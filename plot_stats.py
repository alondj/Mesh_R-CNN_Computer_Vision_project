import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="metrics plotting script")
parser.add_argument("--model", "-m", help="the model we wish plot metrics for", choices=["ShapeNet", "Pix3D"],
                    required=True)
parser.add_argument('--statPath', type=str, required=True, help='the path to the stats file(.st)')

if __name__ == "__main__":
    options = parser.parse_args()
    results = torch.load(options.statPath)

    if options.model == 'ShapeNet':
        batch_times = []
        data_loadings = []
        voxel_losses = []
        edge_losses = []
        normal_losses = []
        chamfer_losses = []
        classifier_losses = []
        epochs = []
        for epoch, meters in results.items():
            epochs.append(epoch)
            batch_times.append(meters['batch_time'].avg)
            data_loadings.append(meters['data_loading'].avg)
            voxel_losses.append(meters['voxel_loss'].avg)
            edge_losses.append(meters['edge_loss'].avg)
            normal_losses.append(meters['normal_loss'].avg)
            chamfer_losses.append(meters['chamfer_loss'].avg)
            classifier_losses.append(meters['loss_classifier'].avg)

        plt.figure(1)
        plt.plot(epochs, batch_times, 'r--', epochs, data_loadings, 'bs')
        plt.ylabel('batch times(red lines),data loading times(blue squers)')
        plt.xlabel('ephocs')
        plt.figure(2)
        plt.plot(epochs, voxel_losses, 'r--')
        plt.ylabel('voxel loss')
        plt.xlabel('ephocs')
        plt.figure(3)
        plt.plot(epochs, edge_losses, 'r--')
        plt.ylabel('edge loss')
        plt.xlabel('ephocs')
        plt.figure(4)
        plt.plot(epochs, normal_losses, 'r--')
        plt.ylabel('normal loss')
        plt.xlabel('ephocs')
        plt.figure(5)
        plt.plot(epochs, chamfer_losses, 'r--')
        plt.ylabel('chamfer loss')
        plt.xlabel('ephocs')
        plt.figure(6)
        plt.plot(epochs, classifier_losses, 'r--')
        plt.ylabel('classifier loss')
        plt.xlabel('ephocs')
        plt.show()

    else:
        batch_times = []
        data_loadings = []
        voxel_losses = []
        edge_losses = []
        normal_losses = []
        chamfer_losses = []
        classifier_losses = []
        box_reg_losses = []
        mask_losses = []
        objectness_losses = []
        rpn_box_reg_loss = []
        epochs = []
        print(results)
        for epoch, meters in results.items():
            epochs.append(epoch)
            batch_times.append(meters['batch_time'].avg)
            data_loadings.append(meters['data_loading'].avg)
            voxel_losses.append(meters['voxel_loss'].avg)
            edge_losses.append(meters['edge_loss'].avg)
            normal_losses.append(meters['normal_loss'].avg)
            chamfer_losses.append(meters['chamfer_loss'].avg)
            classifier_losses.append(meters['loss_classifier'].avg)
            box_reg_losses.append(meters['loss_box_reg'].avg)
            mask_losses.append(meters['loss_mask'].avg)
            objectness_losses.append(meters['loss_objectness'].avg)
            rpn_box_reg_loss.append(meters['loss_rpn_box_reg'].avg)

        plt.figure(1)
        plt.plot(epochs, batch_times, 'r--', epochs, data_loadings, 'bs')
        plt.ylabel('batch times(red lines),data loading times(blue squers)')
        plt.xlabel('ephocs')
        plt.figure(2)
        plt.plot(epochs, voxel_losses, 'r--')
        plt.ylabel('voxel loss')
        plt.xlabel('ephocs')
        plt.figure(3)
        plt.plot(epochs, edge_losses, 'r--')
        plt.ylabel('edge loss')
        plt.xlabel('ephocs')
        plt.figure(4)
        plt.plot(epochs, normal_losses, 'r--')
        plt.ylabel('normal loss')
        plt.xlabel('ephocs')
        plt.figure(5)
        plt.plot(epochs, chamfer_losses, 'r--')
        plt.ylabel('chamfer loss')
        plt.xlabel('ephocs')
        plt.figure(6)
        plt.plot(epochs, classifier_losses, 'r--')
        plt.ylabel('classifier loss')
        plt.xlabel('ephocs')
        plt.figure(7)
        plt.plot(epochs, box_reg_losses, 'r--')
        plt.ylabel('box regression loss')
        plt.xlabel('ephocs')
        plt.figure(8)
        plt.plot(epochs, mask_losses, 'r--')
        plt.ylabel('mask loss')
        plt.xlabel('ephocs')
        plt.figure(9)
        plt.plot(epochs, objectness_losses, 'r--')
        plt.ylabel('objectness loss')
        plt.xlabel('ephocs')
        plt.figure(10)
        plt.plot(epochs, rpn_box_reg_loss, 'r--')
        plt.ylabel('rpn box regression loss')
        plt.xlabel('ephocs')
        plt.show()
