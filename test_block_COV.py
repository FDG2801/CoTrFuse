import torch
from torch.utils.data import DataLoader
from datasets.dataset_modified_COV import Mydataset, test_transform
from tools_mine import Miou_COV as Miou


def test_mertric_here(model, test_imgs, test_masks, save_name,csv):
    test_number = len(test_imgs) #1166 for infection #6788 for covid
    get_csv=csv
    test_ds = Mydataset(get_csv, test_imgs, test_masks, test_transform)
    test_dl = DataLoader(test_ds, batch_size=1, pin_memory=False, num_workers=4, )
    model.load_state_dict(torch.load(save_name + '.pth'))
    model.eval()
    test_dice, test_miou, test_Pre, test_recall, test_F1score, test_pa = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            out = model(inputs)
            predicted = out.argmax(1)
            test_dice += Miou.calculate_mdice(predicted, targets, 2).item()
            test_miou += Miou.calculate_miou(predicted, targets, 2).item()
            test_Pre += Miou.pre(predicted, targets).item()
            test_recall += Miou.recall(predicted, targets).item()
            test_F1score += Miou.F1score(predicted, targets).item()
            test_pa += Miou.Pa(predicted, targets).item()
    average_test_dice = test_dice / 6788
    average_test_miou = test_miou / 6788
    average_test_Pre = test_Pre / 6788
    average_test_recall = test_recall / 6788
    average_test_F1score = test_F1score / 6788
    average_test_pa = test_pa / 6788
    dice, miou, pre, recall, f1_score, pa = \
        '%.4f' % average_test_dice, '%.4f' % average_test_miou, '%.4f' % average_test_Pre, '%.4f' % average_test_recall, '%.4f' % average_test_F1score, '%.4f' % average_test_pa
    return dice, miou, pre, recall, f1_score, pa
