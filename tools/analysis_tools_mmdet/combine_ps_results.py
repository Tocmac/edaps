from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os.path as osp
import os

if __name__ == '__main__':
    rgb_path = '/data/home/wangxu/datasets/cityscapes/leftImg8bit/val'
    ss_gt_path = '/data/home/wangxu/datasets/cityscapes/gtFine/val'
    ps_gt_path = '/data/home/wangxu/datasets/cityscapes/gtFine_panoptic/cityscapes_panoptic_val'
    ps_results_cvrn_path = '/data/home/wangxu/datasets/cityscapes/panoptic_results/cvrn/val_2401231955/results/pans_combined/pan/'
    # cvrn下无子文件夹
    ps_results_edaps_path = '/data/home/wangxu/datasets/cityscapes/panoptic_results/edaps_pred_visuals'

    output_path = '/data/home/wangxu/datasets/cityscapes/panoptic_results/combine'

    filenames_ps_results_cvrn_path = os.listdir(ps_results_cvrn_path)
    n = 0

    for filename_ps_results_cvrn_path in filenames_ps_results_cvrn_path:
        n += 1
        print(n, filename_ps_results_cvrn_path)

        str1 = filename_ps_results_cvrn_path.split('_')
        city, id1, id2 = str1[0], str1[1], str1[2].split('.')[0]
        str2 = filename_ps_results_cvrn_path.split('.')[0]

        img_rgb = mpimg.imread(osp.join(rgb_path, city, str2 + '_leftImg8bit.png'))
        img_ss_gt = mpimg.imread(osp.join(ss_gt_path, city, str2 + '_gtFine_color.png'))
        ps_results_cvrn = mpimg.imread(osp.join(ps_results_cvrn_path, filename_ps_results_cvrn_path))
        ps_results_edaps = mpimg.imread(osp.join(ps_results_edaps_path, city, str2 + '_leftImg8bit_panoptic_seg.png'))
        output_name = osp.join(output_path, filename_ps_results_cvrn_path)

        # img_rgb = mpimg.imread(osp.join(rgb_path, 'frankfurt/frankfurt_000000_000294_leftImg8bit.png'))
        # img_ss_gt = mpimg.imread(osp.join(ss_gt_path, 'frankfurt/frankfurt_000000_000294_gtFine_color.png'))
        # ps_results_cvrn = mpimg.imread(osp.join(ps_results_cvrn_path, 'frankfurt_000000_000294.png'))
        # ps_results_edaps = mpimg.imread(osp.join(ps_results_edaps_path, 'frankfurt/frankfurt_000000_000294_leftImg8bit_panoptic_seg.png'))

        fig, axs = plt.subplots(2, 2, figsize=(20.48, 10.24))

        axs[0, 0].axis('off')
        axs[0, 1].axis('off')
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')

        axs[0, 0].set_title('RGB')
        axs[0, 1].set_title('GT')
        axs[1, 0].set_title('Baseline')
        axs[1, 1].set_title('Ours')

        axs[0, 0].imshow(img_rgb)
        axs[0, 1].imshow(img_ss_gt)
        axs[1, 0].imshow(ps_results_cvrn)
        axs[1, 1].imshow(ps_results_edaps)

        # plt.show()
        fig.savefig(output_name)  # save the figure to file
        # mmcv.imwrite(panoptic_pred, out_vis_fname_pan_Seg)
        plt.close(fig)