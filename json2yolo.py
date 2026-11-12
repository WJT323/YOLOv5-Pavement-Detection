import json
import os
# 文件代码功能：是将labelme打标产生的json文件转换为yolo所能识别的标准格式txt。
# json中共出现19个类别（已全量扫描 highway_label/jsons 统计得到）
name2id = {
    '坑槽': 0,
    '松散': 1,
    '横向裂纹': 2,
    '纵向裂纹': 3,
    '网状裂纹': 4,
    '横向修补': 5,
    '纵向修补': 6,
    '块状修补': 7,
    '网状修补': 8,
    '标线模糊': 9,
    '抛洒物': 10,
    '锥形桶': 11,
    '圆形桶': 12,
    '人': 13,
    '方形雨水盖': 14,
    '圆形雨水盖': 15,
    '圆形井盖': 16,
    '护栏损坏': 17,
}
# 标注中的笔误别名：'纵向裂缝' 归并为 '纵向裂纹'（如不想合并，删掉此行即可）
label_alias = {'纵向裂缝': '纵向裂纹'}


def convert(img_size, box):  # 对标签坐标做单位化（归一化到0~1）
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def decode_json(json_floder_path, json_name):
    # 转换好的标签路径
    txt_name = os.path.join(txt_dir, json_name[0:-5] + '.txt')
    txt_file = open(txt_name, 'w')
    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    img_w = data['imageWidth']
    img_h = data['imageHeight']
    for i in data['shapes']:
        label_name = i['label']
        label_name = label_alias.get(label_name, label_name)
        if (i['shape_type'] == 'rectangle'):
            # labelme矩形两点顺序不固定，取min/max保证左上、右下
            x1 = min(i['points'][0][0], i['points'][1][0])
            y1 = min(i['points'][0][1], i['points'][1][1])
            x2 = max(i['points'][0][0], i['points'][1][0])
            y2 = max(i['points'][0][1], i['points'][1][1])
            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')
    txt_file.close()


if __name__ == "__main__":
    # json文件的路径（基于本脚本所在目录定位，不受运行目录影响）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_floder_path = os.path.join(base_dir, 'highway_label', 'jsons')
    # txt输出的路径
    txt_dir = os.path.join(base_dir, 'highway_label', 'labels')
    os.makedirs(txt_dir, exist_ok=True)
    json_names = [f for f in os.listdir(json_floder_path) if f.endswith('.json')]
    for n, json_name in enumerate(json_names):
        decode_json(json_floder_path, json_name)
        if (n + 1) % 500 == 0:
            print('已转换 %d/%d' % (n + 1, len(json_names)))
    print('完成，共转换 %d 个文件，输出目录: %s' % (len(json_names), txt_dir))
