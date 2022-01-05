import json
import random
import string


def via_json(video_filename, description=None):
    """
    Create a dictionary to be saved as VIA json annotation file.

    :param video_filename: relative video filename_or_buffer
    :param description: annotation description
    :return: dict
    """
    json_template = """
    {
      "project": {
        "pid": "__VIA_PROJECT_ID__",
        "rev": "__VIA_PROJECT_REV_ID__",
        "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
        "pname": "",
        "creator": "",
        "created": 1573815248007,
        "vid_list": [
          "1"
        ]
      },
      "config": {
        "file": {
          "loc_prefix": {
            "1": "",
            "2": "",
            "3": "",
            "4": ""
          }
        },
        "ui": {
          "file_content_align": "center",
          "file_metadata_editor_visible": true,
          "spatial_metadata_editor_visible": true,
          "spatial_region_label_attribute_id": ""
        }
      },
      "attribute": {},
      "file": {
        "1": {
          "fid": "1",
          "fname": "",
          "type": 4,
          "loc": 1,
          "src": ""
        }
      },
      "metadata": {},
      "view": {
        "1": {
          "fid_list": [
            "1"
          ]
        }
      }
    }
    """
    json_out = json.loads(json_template)
    json_out["file"]["1"]["fname"] = video_filename
    if description is not None:
        json_out["project"]["pname"] = description
    return json_out


def attribute(
    name, anchor_id, attribute_type, desc=None, options=None, default_option_id=None
):
    """
    Create VIA attribute.

    :param name: str, attribute name
    :param anchor_id:
      'FILE1_Z0_XY0':'Attribute of a File (e.g. image caption)',
      'FILE1_Z0_XY1':'Spatial Region in an Image (e.g. bounding box of an object)',
      'FILE1_Z0_XYN':'__FUTURE__',   // File region composed of multiple disconnected regions
      'FILE1_Z1_XY0':'__FUTURE__',   // Time marker in video or audio (e.g tongue clicks, speaker diarisation)
      'FILE1_Z1_XY1':'Spatial Region in a Video Frame (e.g. bounding box of an object)',
      'FILE1_Z1_XYN':'__FUTURE__',   // A video frame region composed of multiple disconnected regions
      'FILE1_Z2_XY0':'Temporal Segment in Video or Audio (e.g. video segment containing an actor)',
      'FILE1_Z2_XY1':'__FUTURE__',   // A region defined over a temporal segment
      'FILE1_Z2_XYN':'__FUTURE__',   // A temporal segment with regions defined for start and end frames
      'FILE1_ZN_XY0':'__FUTURE__',   // ? (a possible future use case)
      'FILE1_ZN_XY1':'__FUTURE__',   // ? (a possible future use case)
      'FILE1_ZN_XYN':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z0_XY0':'Attribute of a Group of Files (e.g. given two images, which is more sharp?)',
      'FILEN_Z0_XY1':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z0_XYN':'__FUTURE__',   // one region defined for each file (e.g. an object in multiple views)
      'FILEN_Z1_XY0':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z1_XY1':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z1_XYN':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z2_XY0':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z2_XY1':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_Z2_XYN':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_ZN_XY0':'__FUTURE__',   // one timestamp for each video or audio file (e.g. for alignment)
      'FILEN_ZN_XY1':'__FUTURE__',   // ? (a possible future use case)
      'FILEN_ZN_XYN':'__FUTURE__',   // a region defined in a video frame of each video
    :param attribute_type: 'TEXT', 'CHECKBOX', 'RADIO', 'SELECT', 'IMAGE'
    :param desc:
    :param options:
    :param default_option_id:
    :return:
    """
    VIA_ATTRIBUTE_TYPE = {"TEXT": 1, "CHECKBOX": 2, "RADIO": 3, "SELECT": 4, "IMAGE": 5}
    if desc is None:
        desc = ""
    if options is None:
        options = []
    if default_option_id is None:
        default_option_id = ""
    return {
        "aname": name,
        "anchor_id": anchor_id,
        "type": VIA_ATTRIBUTE_TYPE[attribute_type],
        "desc": desc,
        "options": {i: val for i, val in enumerate(options, start=1)},
        "default_option_id": default_option_id,
    }


def metadata(time, shape_type=None, coords=None, attributes=None):
    """
    Create single VIA metadata record.

    :param time: time in seconds
    :param shape_type: see VIA_RSHAPE
    :param coords: RECTANGLE: [x,y,w,h], LINE: [x1,y1,x2,y2], ...
    :param attributes: {attribute id: attribute value index, ...}
    :return: metadata dictionary
    """
    VIA_RSHAPE = {
        "POINT": 1,
        "RECTANGLE": 2,
        "CIRCLE": 3,
        "ELLIPSE": 4,
        "LINE": 5,
        "POLYLINE": 6,
        "POLYGON": 7,
        "EXTREME_RECTANGLE": 8,
        "EXTREME_CIRCLE": 9,
    }
    if shape_type is None:
        assert coords is None
        xy = []
        assert (
            attributes
        ), "temporal annotation requires an FILE1_Z2_XY0 attribute value, e.g. {'1': '_DEFAULT'}"
    else:
        xy = [VIA_RSHAPE[shape_type]] + coords
    if attributes is None:
        av = {}
    else:
        av = attributes
    if not isinstance(time, (list, tuple)):
        time = [time]
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"1_{random_id}", {"vid": "1", "flg": 0, "z": time, "xy": xy, "av": av}


if __name__ == "__main__":
    json_out = via_json("video.mp4", "hunting for a rectangle ground truth")
    color_options = ["red", "green"]
    json_out["attribute"] = {
        1: attribute("locations", "FILE1_Z2_XY0", "TEXT"),
        2: attribute("color", "FILE1_Z1_XY1", "RADIO", options=color_options),
    }
    for time_s in range(100):
        json_out["metadata"].update(
            (
                metadata(
                    time_s,
                    "RECTANGLE",
                    [100 + time_s * 10, 100, 150, 150],
                    {"2": "1" if time_s % 2 == 0 else "2"},
                ),
            )
        )  # second attribute, '1' is red, '2' is green
    with open("rectangles.json", "w") as fw:
        json.dump(json_out, fw)
