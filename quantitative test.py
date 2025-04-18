import utils.metrics as Metrics
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ResultMetrics:
    def __init__(self, path, ct_path, mri_path, name):
        self.path = path
        self.ct_path = ct_path
        self.mri_path = mri_path
        self.img_list = os.listdir(path)
        self.img_data = []
        self.name = name
        for img in self.img_list:
            ct_img = cv2.cvtColor(cv2.imread(self.ct_path + img), cv2.COLOR_BGR2GRAY)
            mri_img = cv2.cvtColor(cv2.imread(self.mri_path + img), cv2.COLOR_BGR2GRAY)
            fusion_img = cv2.cvtColor(cv2.imread(self.path + img), cv2.COLOR_BGR2GRAY)
            self.img_data.append([ct_img, mri_img, fusion_img])

    def calc_ssim(self):
        ssim = []
        for image in self.img_data:
            ct_img, mri_img, fusion_img = image
            ct_ssim = Metrics.calculate_ssim(fusion_img, ct_img)
            mri_ssim = Metrics.calculate_ssim(fusion_img, mri_img)
            ssim.append([ct_ssim, mri_ssim])
        return np.array(ssim)

    def calc_mi(self):
        mi = []
        for image in self.img_data:
            ct_img, mri_img, fusion_img = image
            ct_mi = Metrics.calculate_mi(fusion_img, ct_img)
            mri_mi = Metrics.calculate_mi(fusion_img, mri_img)
            mi.append([ct_mi, mri_mi])
        return np.array(mi)

    def calc_vif(self):
        vif = []
        for image in self.img_data:
            ct_img, mri_img, fusion_img = image
            ct_vif = Metrics.calculate_vif(fusion_img, ct_img)
            mri_vif = Metrics.calculate_vif(fusion_img, mri_img)
            vif.append([ct_vif, mri_vif])
        return np.array(vif)

    def calc_qabf(self):
        qabf = []
        for image in self.img_data:
            ct_img, mri_img, fusion_img = image
            q = Metrics.calculate_qabf(ct_img, mri_img, fusion_img, 1)
            qabf.append(q)
        return np.array(qabf)

    def calc_psnr(self):
        psnr = []
        for image in self.img_data:
            ct_img, mri_img, fusion_img = image
            ct_psnr = Metrics.calculate_psnr(ct_img, fusion_img)
            mri_psnr = Metrics.calculate_psnr(mri_img, fusion_img)
            psnr.append([ct_psnr, mri_psnr])
        return np.array(psnr)

    def calc_scc(self):
        scc = []
        for image in self.img_data:
            ct_img, mri_img, fusion_img = image
            ct_scc = Metrics.calculate_scc(ct_img, fusion_img)
            mri_scc = Metrics.calculate_scc(mri_img, fusion_img)
            scc.append([ct_scc, mri_scc])
        return np.array(scc)



if __name__ == "__main__":
    path_prefix = 'quantitative_test_data/'
    path_ct_prefix = 'quantitative_test_data/eval/CT/'
    path_mri_prefix = 'quantitative_test_data/eval/MRI/'
    if os.path.exists('quantitative_test_data/eval'):
        src_ct_list = os.listdir('quantitative_test_data/eval/CT')
        src_mri_list = os.listdir('quantitative_test_data/eval/MRI')
        assert src_ct_list == src_mri_list, "CT and MRI data mismatch"

    result = os.listdir('quantitative_test_data')
    result.remove('eval')
    result_instance = []
    for r in result:
        result_instance.append(ResultMetrics(path_prefix + r + '/', path_ct_prefix, path_mri_prefix, r))

    paint_pos = 0
    bar_width = 1
    # Set up color
    cmap = plt.get_cmap('tab10')
    instance_num = len(result_instance)
    colors = [cmap(i) for i in range(instance_num)]
    # SSIM comparison
    average_ssim = []
    count = 0
    for instance in result_instance:
        ssim = instance.calc_ssim()
        average_ssim.append(np.mean(ssim))
        plt.bar(paint_pos, np.mean(ssim), width=bar_width, label=instance.name, color=colors[count % instance_num], align='edge')
        paint_pos += bar_width
        count += 1
    paint_pos += 5 * bar_width

    # MI comparison
    average_mi = []
    for instance in result_instance:
        mi = instance.calc_mi()
        average_mi.append(np.mean(mi))
        plt.bar(paint_pos, np.mean(mi), width=bar_width, color=colors[count % instance_num], align='edge')
        paint_pos += bar_width
        count += 1
    paint_pos += 5 * bar_width

    # VIF comparison
    average_vif = []
    for instance in result_instance:
        vif = instance.calc_vif()
        average_vif.append(np.mean(vif))
        plt.bar(paint_pos, np.mean(vif), width=bar_width, color=colors[count % instance_num], align='edge')
        paint_pos += bar_width
        count += 1
    paint_pos += 5 * bar_width

    # QABF comparison
    average_qabf = []
    for instance in result_instance:
        qabf = instance.calc_qabf()
        average_qabf.append(np.mean(qabf))
        plt.bar(paint_pos, np.mean(qabf), width=bar_width, color=colors[count % instance_num], align='edge')
        paint_pos += bar_width
        count += 1
    paint_pos += 5 * bar_width

    # PSNR comparison
    average_psnr = []
    for instance in result_instance:
        psnr = instance.calc_psnr()
        average_psnr.append(np.mean(psnr))
        plt.bar(paint_pos, np.mean(psnr), width=bar_width, color=colors[count % instance_num], align='edge')
        paint_pos += bar_width
        count += 1
    paint_pos += 5 * bar_width

    # SCC comparison
    average_scc = []
    for instance in result_instance:
        scc = instance.calc_scc()
        average_scc.append(np.mean(scc))
        plt.bar(paint_pos, np.mean(scc), width=bar_width, color=colors[count % instance_num], align='edge')
        paint_pos += bar_width
        count += 1
    paint_pos += 5 * bar_width

    plt.xlabel('Metrics')
    plt.xlim(left=0)
    plt.xticks([1, 8, 15, 22, 29, 36],['SSIM', 'MI', 'VIF', 'QABF', 'PSNR', 'SCC'])
    plt.ylabel('Value')
    plt.title('Quantitative Metrics Comparison')
    plt.legend()
    plt.show()
