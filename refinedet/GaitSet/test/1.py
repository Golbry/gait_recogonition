import os
path_root = 'E:\\code\\GaitSet-master\\dataset\\001'
dirs = os.listdir(path_root)

# 输出所有文件和文件夹
for file in dirs:
    path_test = os.path.join(path_root, file)
    #print(path)
    print(path_test)
#也可以直接进行将多个路径组合后返回
p = os.path.join('F:\\data\\input','test')
print(p)

