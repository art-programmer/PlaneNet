import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os

from utils import *
from options import parse_args

from models.planenet import PlaneNet
from models.modules import *

from datasets.plane_dataset import PlaneDataset

def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    dataset = PlaneDataset(options, split='train', random=True)
    dataset_test = PlaneDataset(options, split='test', random=False)

    print('the number of images', len(dataset), len(dataset_test))    

    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=16)

    model = PlaneNet(options)
    model.cuda()
    model.train()

    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/checkpoint.pth'):
        print('restore')
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        pass

    plane_criterion = torch.nn.MSELoss(reduce=False)    
    segmentation_criterion = torch.nn.CrossEntropyLoss()
    depth_criterion = torch.nn.MSELoss(reduce=False)    
    
    if options.task == 'test':
        testOneEpoch(options, model, plane_criterion, segmentation_criterion, depth_criterion, dataset_test)
        exit(1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = options.LR)
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))
        pass

    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
        for sampleIndex, sample in enumerate(data_iterator):
            image_inp, planes_gt, segmentation_gt, depth_gt, metadata, numbers = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda()

            optimizer.zero_grad()
            planes_pred, segmentation_pred, non_plane_depth_pred = model(image_inp)

            distances = torch.norm(planes_gt.unsqueeze(2) - planes_pred.unsqueeze(1), dim=-1)
            W = distances.max() - distances.transpose(1, 2)
            mapping = torch.stack([assignmentModule(W[batchIndex]) for batchIndex in xrange(len(distances))], dim=0)

            mapping = oneHotModule(mapping.view(-1), depth=int(planes_pred.shape[1])).view((int(mapping.shape[0]), int(mapping.shape[1]), -1))
            planes_pred_shuffled = torch.matmul(mapping, planes_pred)
            segmentation_pred_shuffled = torch.matmul(mapping, segmentation_pred[:, :-1].view((int(segmentation_pred.shape[0]), options.numOutputPlanes, -1))).view(segmentation_pred[:, :-1].shape)
            segmentation_pred_shuffled = torch.cat([segmentation_pred_shuffled, segmentation_pred[:, -1:]], dim=1)

            validMask = (torch.arange(int(planes_gt.shape[1]), dtype=torch.int64).cuda() < numbers.unsqueeze(-1)).float()
            
            plane_loss = torch.sum(plane_criterion(planes_pred_shuffled, planes_gt) * validMask.unsqueeze(-1)) / torch.sum(validMask)
            segmentation_loss = segmentation_criterion(segmentation_pred_shuffled, segmentation_gt)

                        
            segmentation_pred_shuffled = torch.nn.Softmax(dim=1)(segmentation_pred_shuffled)
            plane_depths = calcPlaneDepthsModule(options.outputWidth, options.outputHeight, planes_pred_shuffled, metadata[0])
            all_depths = torch.cat([plane_depths.transpose(2, 3).transpose(1, 2), non_plane_depth_pred], dim=1)
            depth_mask = ((depth_gt > 1e-4) & (depth_gt < MAX_DEPTH)).float()
            depth_loss = torch.mean(torch.pow(all_depths - depth_gt.unsqueeze(1), 2) * depth_mask.unsqueeze(1) * segmentation_pred_shuffled) * 10
            depth_pred = torch.sum(all_depths * segmentation_pred_shuffled, 1) * depth_mask

            segmentation_pred_shuffled = segmentation_pred_shuffled.max(1)[1]
            #depth_pred = calcDepthModule(options.outputWidth, options.outputHeight, planes_pred, segmentation_pred, non_plane_depth_pred, metadata[0])
            #depth_loss = depth_criterion(depth_pred, depth_gt)

            depth_gt = calcDepthModule(options.outputWidth, options.outputHeight, planes_gt, oneHotModule(segmentation_gt, depth=options.numOutputPlanes + 1).transpose(2, 3).transpose(1, 2), depth_gt.unsqueeze(1), metadata[0])
            
            
            #print(torch.round(adjacency_pred_shuffled[0]).long())
            #print(adjacency_gt[0])
            
            losses = [plane_loss, segmentation_loss, depth_loss]
            loss = sum(losses)

            loss_values = [l.data.item() for l in losses]
            epoch_losses.append(loss_values)
            status = str(epoch + 1) + ' loss: '
            for l in loss_values:
                status += '%0.5f '%l
                continue
            data_iterator.set_description(status)
            loss.backward()
            optimizer.step()

            if sampleIndex % 500 == 0:
                visualizeBatchPlanes(options, image_inp.detach().cpu().numpy(), numbers.long().detach().cpu().numpy(), [('gt', {'plane': planes_gt.detach().cpu().numpy(), 'segmentation': segmentation_gt.detach().cpu().numpy(), 'depth': depth_gt.detach().cpu().numpy()}), ('pred', {'plane': planes_pred_shuffled.detach().cpu().numpy(), 'segmentation': segmentation_pred_shuffled.detach().cpu().numpy(), 'depth': depth_pred.detach().cpu().numpy()})])
                if options.visualizeMode == 'debug':
                    exit(1)
                    pass
            continue
        print('loss', np.array(epoch_losses).mean(0))
        if True:
            torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
            pass

        testOneEpoch(options, model, plane_criterion, segmentation_criterion, depth_criterion, dataset_test)        
        continue

    #test(options)
    return

def testOneEpoch(options, model, plane_criterion, segmentation_criterion, depth_criterion, dataset):
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=4)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):
        image_inp, planes_gt, segmentation_gt, depth_gt, metadata, numbers = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda()

        planes_pred, segmentation_pred, non_plane_depth_pred = model(image_inp)

        distances = torch.norm(planes_gt.unsqueeze(2) - planes_pred.unsqueeze(1), dim=-1)
        W = distances.max() - distances.transpose(1, 2)
        mapping = torch.stack([assignmentModule(W[batchIndex]) for batchIndex in xrange(len(distances))], dim=0)

        mapping = oneHotModule(mapping.view(-1), depth=int(planes_pred.shape[1])).view((int(mapping.shape[0]), int(mapping.shape[1]), -1))
        planes_pred_shuffled = torch.matmul(mapping, planes_pred)
        segmentation_pred_shuffled = torch.matmul(mapping, segmentation_pred[:, :-1].view((int(segmentation_pred.shape[0]), options.numOutputPlanes, -1))).view(segmentation_pred[:, :-1].shape)
        segmentation_pred_shuffled = torch.cat([segmentation_pred_shuffled, segmentation_pred[:, -1:]], dim=1)

        validMask = (torch.arange(int(planes_gt.shape[1]), dtype=torch.int64).cuda() < numbers.unsqueeze(-1)).float()

        plane_loss = torch.sum(plane_criterion(planes_pred_shuffled, planes_gt) * validMask.unsqueeze(-1)) / torch.sum(validMask)
        segmentation_loss = segmentation_criterion(segmentation_pred_shuffled, segmentation_gt)


        segmentation_pred_shuffled = torch.nn.Softmax(dim=1)(segmentation_pred_shuffled)
        plane_depths = calcPlaneDepthsModule(options.outputWidth, options.outputHeight, planes_pred_shuffled, metadata[0])
        all_depths = torch.cat([plane_depths.transpose(2, 3).transpose(1, 2), non_plane_depth_pred], dim=1)
        depth_mask = ((depth_gt > 1e-4) & (depth_gt < MAX_DEPTH)).float()
        depth_loss = torch.mean(torch.pow(all_depths - depth_gt.unsqueeze(1), 2) * depth_mask.unsqueeze(1) * segmentation_pred_shuffled) * 10
        depth_pred = torch.sum(all_depths * segmentation_pred_shuffled, 1) * depth_mask

        #depth_pred = calcDepthModule(options.outputWidth, options.outputHeight, planes_pred, segmentation_pred, non_plane_depth_pred, metadata[0])
        #depth_loss = depth_criterion(depth_pred, depth_gt)

        #depth_gt = calcDepthModule(options.outputWidth, options.outputHeight, planes_gt, oneHotModule(segmentation_gt, depth=options.numOutputPlanes + 1).transpose(2, 3).transpose(1, 2), depth_gt.unsqueeze(1), metadata[0])


        segmentation_pred = segmentation_pred.max(1)[1]
        segmentation_pred_shuffled = segmentation_pred_shuffled.max(1)[1]        
        #print(torch.round(adjacency_pred_shuffled[0]).long())
        #print(adjacency_gt[0])

        losses = [plane_loss, segmentation_loss, depth_loss]
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        if sampleIndex % 500 == 0:
            visualizeBatchPlanes(options, image_inp.detach().cpu().numpy(), numbers.long().detach().cpu().numpy(), [('gt', {'plane': planes_gt.detach().cpu().numpy(), 'segmentation': segmentation_gt.detach().cpu().numpy(), 'depth': depth_gt.detach().cpu().numpy()}), ('pred', {'plane': planes_pred_shuffled.detach().cpu().numpy(), 'segmentation': segmentation_pred_shuffled.detach().cpu().numpy(), 'depth': depth_pred.detach().cpu().numpy()})], prefix='test')
            if options.visualizeMode == 'debug':
                exit(1)
                pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))

    model.train()
    return

def visualizeBatchPlanes(options, image_inp, numbers, dicts, indexOffset=0, prefix=''):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    image_inp = image_inp.transpose((0, 2, 3, 1))
    for batchIndex in range(len(image_inp)):
        if prefix == '':
            filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_input.png'
        else:
            filename = options.test_dir + '/' + prefix + '_' + str(indexOffset + batchIndex) + '_input.png'
            pass

        inputImage = ((image_inp[batchIndex] + MEAN_STD[0]) * 255).astype(np.uint8)        
        cv2.imwrite(filename, inputImage)
        for name, result_dict in dicts:
            #image = inputImage.copy()
            segmentationImage = drawSegmentationImage(result_dict['segmentation'][batchIndex], blackIndex=options.numOutputPlanes)
            cv2.imwrite(filename.replace('input', 'segmentation_' + name), segmentationImage)
            depthImage = drawDepthImage(result_dict['depth'][batchIndex])
            cv2.imwrite(filename.replace('input', 'depth_' + name), depthImage)
            continue
        continue
    return

if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'planenet'
    #args.keyname += '_' + args.dataset

    if args.suffix != '':
        args.keyname += '_' + suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
