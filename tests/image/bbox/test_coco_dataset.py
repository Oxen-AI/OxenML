
import sys, os
import pathlib

# Need to add the oxen dir to test paths
oxen_dir = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(oxen_dir))

from oxen.image.bounding_box.annotations.oxen_bounding_box import OxenBoundingBox
from oxen.image.bounding_box import CocoBoundingBoxDataset

def test_load_coco_instances():
    dataset = CocoBoundingBoxDataset(
        annotation_file="tests/data/coco_bbox_categories_small.json"
    )
    assert dataset.num_inputs() == 4
    
    file = "000000397133.jpg"
    file_annotations = dataset.get_annotations(file)
    assert len(file_annotations) == 2

    first_annotation = file_annotations[0]
    assert(first_annotation.min_x == 217.62)
    assert(first_annotation.min_y == 240.54)
    assert(first_annotation.width == 38.99)
    assert(first_annotation.height == 57.75)

def test_convert_coco_to_tsv():
    dataset = CocoBoundingBoxDataset(
        annotation_file="tests/data/coco_bbox_categories_small.json"
    )
    
    base_dir = "test"
    outfile = "/tmp/test_coco_output.tsv"
    dataset.write_tsv(base_dir, outfile=outfile)
    
    with open(outfile) as f:
        lines = f.readlines()
        assert(lines[0] == "file\tmin_x\tmin_y\twidth\theight\n")
        assert(lines[1] == "test/000000397133.jpg\t217.62\t240.54\t38.99\t57.75\n")
    
    os.remove(outfile)

def test_convert_coco_to_csv():
    dataset = CocoBoundingBoxDataset(
        annotation_file="tests/data/coco_bbox_categories_small.json"
    )
    
    base_dir = "test"
    outfile = "/tmp/test_coco_output.csv"
    dataset.write_csv(base_dir, outfile=outfile)
    
    with open(outfile) as f:
        lines = f.readlines()
        assert(lines[0] == "file,min_x,min_y,width,height\n")
        assert(lines[1] == "test/000000397133.jpg,217.62,240.54,38.99,57.75\n")
    
    os.remove(outfile)
    
