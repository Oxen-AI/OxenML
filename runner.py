
import argparse
# from tools.keypoints import annotate as keypoints_annotate
# from tools.keypoints import plot as keypoints_plot
# from tools.keypoints import resize as keypoints_resize
from tools.bounding_box import convert as bounding_box_convert
from typing import Callable, List
import sys

def print_help(tools: dict[str, Callable]):
    print("Usage: python runner.py <tool> <args>\n\nRunner can run a suite of scripts including:")
    for tool_name in tools.keys():
        print(f"\t{tool_name}")
    exit()

def main():
    # Mapping of tool names to functions they will call
    tools = {
        # "keypoints_resize": keypoints_resize,
        # "keypoints_annotate": keypoints_annotate,
        # "keypoints_plot": keypoints_plot,

        "bbox_convert": bounding_box_convert,
    }
    
    if len(sys.argv) < 2:
        print_help(tools)

    tool_name = sys.argv[1]
    if "-h" == tool_name or "--help" == tool_name:
        print_help(tools)

    if tool_name in tools:
        args = sys.argv[2:len(sys.argv)]
        tools[tool_name](args)
    else:
        print(f"Unknown tool [{tool_name}]")
        print_help(tools)


if __name__ == '__main__':
    main()