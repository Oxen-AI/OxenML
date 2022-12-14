import argparse

# from tools.keypoints import annotate as keypoints_annotate
# from tools.keypoints import plot as keypoints_plot
# from tools.keypoints import resize as kp_resize
from tools.keypoints import convert as kp_convert
from tools.bounding_box import convert as bb_convert
from tools.bounding_box import resize as bb_resize
from tools.bounding_box import plot as bb_plot
from tools.bounding_box import predict as bb_predict
from tools.text_classification import convert as tc_convert
from tools.text_classification import train as tc_train
from tools.text_classification import predict as tc_predict
from typing import Callable, List
import sys


def print_help(tools: dict[str, Callable]):
    print(
        "Usage: python runner.py <tool> <args>\n\nRunner can run a suite of scripts including:"
    )
    for tool_name in tools.keys():
        print(f"\t{tool_name}")
    exit()


def main():
    # Mapping of tool names to functions they will call
    tools = {
        # "kp_resize": kp_resize,
        # "keypoints_annotate": keypoints_annotate,
        # "keypoints_plot": keypoints_plot,
        "kp_convert": kp_convert,
        "bbox_convert": bb_convert,
        "bbox_resize": bb_resize,
        "bbox_plot": bb_plot,
        "bbox_predict": bb_predict,
        "tc_convert": tc_convert,
        "tc_train": tc_train,
        "tc_predict": tc_predict,
    }

    if len(sys.argv) < 2:
        print_help(tools)

    tool_name = sys.argv[1]
    if "-h" == tool_name or "--help" == tool_name:
        print_help(tools)

    if tool_name in tools:
        args = sys.argv[2 : len(sys.argv)]
        tools[tool_name](args)
    else:
        print(f"Unknown tool [{tool_name}]")
        print_help(tools)


if __name__ == "__main__":
    main()
