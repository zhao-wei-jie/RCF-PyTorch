from scipy.io import savemat
from glob import glob
import cv2,numpy as np
import json
import matplotlib.pyplot as plt
import os.path as osp
# alist=glob('results/RCF20220112_1910/epoch1-test/*ss.png')
# print(len(alist))
# key = "result"
# for i in alist:
#     image=cv2.imread(i,cv2.IMREAD_GRAYSCALE).astype(np.float_)/255
#     print(image.shape)
#     savemat(i.replace('.png','.mat'), {key: image})
evallist=[]
path='results/RCF20220113_1940/test.log'
with open(path) as f:
    
    for idx,line in enumerate(f):
        # print(idx)
        if(idx%2==1): 
            # print(idx)
            continue
        eval={}
        line = line[:-1]
        # a=json.loads(res)
        line=line.strip('{').strip('}')

        
        # print(line)
        
        alist=line.split(',')
        for idx,i in enumerate(alist):
            
            i=i.split(':')
            try:
                eval[i[0].strip(' ').strip('\'')]=float(i[1])
            except ValueError:
                continue
                eval[i[0].strip(' ').strip('\'')]=i[1].strip(' ').strip('\'')
        
        # print(eval)
        
        evallist.append(eval)
        # break
# print(evallist)
plot_epochs=range(len(evallist))
iou_values=[]
f1_values=[]
for i in evallist:
    # tmp=[i['IoU.pl'],i['Fscore.pl']]
    iou_values.append(i['IoU.pl'])
    try:
        f1_values.append(i['Fscore.pl'])
    except:
        f1_values.append(i['f1'])
iou_values.sort()
f1_values.sort()
ax = plt.gca()
# label = legend[i * num_metrics + j]
# if metric in ['mIoU', 'mAcc', 'aAcc']:
ax.set_xticks(plot_epochs)
plt.xlabel('epoch')
plt.plot(plot_epochs, iou_values, label='iou')
plt.plot(plot_epochs, f1_values, label='f1')
plt.text(plot_epochs[np.argmax(iou_values)], max(iou_values), max(iou_values))#写出最高点值
plt.text(plot_epochs[np.argmax(f1_values)], max(f1_values), max(f1_values))#写出最高点值
plt.legend()
plt.xticks(rotation=45)#x坐标数字倾斜角度
plt.grid()#画出网格
plt.xticks((range(1,len(evallist)+1,10)))
# else:
#     plt.xlabel('iter')
#     plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
#     plt.text(plot_iters[np.argmax(plot_values)], max(plot_values), max(plot_values))#写出最高点值
#     ax.set_xticks(plot_iters)#x坐标显示的数值
#     # plt.xticks(range(0,max(plot_iters),8000))
#     plt.legend()
#     # plt.subplots_adjust(bottom=0.2) #设置底部宽度
#     plt.xticks(rotation=45)#x坐标数字倾斜角度
#     plt.grid()#画出网格
# if args.title is not None:
#     plt.title(args.title)

out=osp.dirname(path)
print(f'save curve to: {out}')
plt.savefig(osp.join(out,'fig'))
plt.cla()