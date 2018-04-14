import os
def assert_dispatch_path(path):
    assert os.path.exists(path),\
            "路径{}不存在".format(path)

def assert_imagelist(image_list):
    assert image_list,\
            "image_list不能为空"