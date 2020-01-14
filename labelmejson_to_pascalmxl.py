import os
import glob
import json
import xml.dom.minidom as md
#import xml.etree.ElementTree as ET
from lxml import etree as ET
from tqdm import tqdm


XML_ATTRIBUTE_BASE = 'annotation'
XML_ATTRIBUTE_FILENAME = 'filename'
XML_ATTRIBUTE_SOURCE = 'source'
XML_ATTRIBUTE_DATABASE = 'database'
XML_ATTRIBUTE_IMGSIZE = 'size'
XML_ATTRIBUTE_IMG_H = 'width'
XML_ATTRIBUTE_IMG_W = 'height'
XML_ATTRIBUTE_IMG_C = 'depth'
XML_ATTRIBUTE_SEGMENTED = 'segmented'
XML_ATTRIBUTE_OBJECT = 'object'
XML_ATTRIBUTE_LAVEL = 'name'
XML_ATTRIBUTE_POSE = 'pose'
XML_ATTRIBUTE_TRUNCATED = 'truncated'
XML_ATTRIBUTE_DIFFICULT = 'difficult'
XML_ATTRIBUTE_BOX = 'bndbox'
XML_ATTRIBUTE_X_MIN = 'xmin'
XML_ATTRIBUTE_Y_MIN = 'ymin'
XML_ATTRIBUTE_X_MAX = 'xmax'
XML_ATTRIBUTE_Y_MAX = 'ymax'

XML_DEFAULT_DATABASE = 'Unknown'
XML_DEFAULT_SEGMENTED = '0'
XML_DEFAULT_POSE = 'Unspecified'
XML_DEFAULT_TRUNCATED = '0'
XML_DEFAULT_DIFFICULT = '0'


"""
Pascal VOC mxl Samples :

<annotation>
    <filename>DJI_00120016.jpg</filename>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>3840</width>
        <height>2160</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>3538</xmin>
            <ymin>2006</ymin>
            <xmax>3604</xmax>
            <ymax>2096</ymax>
        </bndbox>
    </object>
</annotation>
"""




def labelmejson_to_pascalmxl_for_dir(target_dir, include__sub_dirs=True):
    #files = glob.glob(os.path.join(target_dir, '*.json'))
    #if include__sub_dirs:
    #    files = glob.glob(os.path.join(target_dirs, '*', '*.json'))

    dirs = [os.path.basename(target_dir)]
    if include__sub_dirs:
        dirs = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]

    dirs_count = len(dirs)
    for k, d in enumerate(dirs):
        files = glob.glob(os.path.join(target_dir, d, '*.json'))
        files.sort()
        print('### %03d / %03d ' % (k + 1, dirs_count), d, ' : ', len(files), ' files')
        pbar = tqdm(total=len(files), desc='convert', unit=' Files')
        for f in files:
            fname, _ = os.path.splitext(os.path.basename(f))

            pascalxml = labelmejson_to_pascalmxl(f)

            xml_tree = ET.ElementTree(pascalxml)
            xml_tree.write(os.path.join(target_dir, d, fname + '.xml'), encoding='UTF-8', pretty_print=True)

            """
            xml_tree = ET.Element(pascalxml)
            document = md.parseString(ET.tostring(xml_tree, 'utf-8'))
            xml_file = open(os.path.join(target_dir, d, fname + '.xml'), 'w')
            document.writexml(xml_file, encoding='utf-8', newl='\n', indent='', addindent='\t')
            xml_file.close()
            """
            pbar.update(1)
        pbar.close()
        print('')
    return True


def labelmejson_to_pascalmxl(labelmejson):
    jf = json.load(open(labelmejson))
    #image_name_base, _ = os.path.splitext(os.path.basename(labelmejson))
    fname = jf['imagePath']
    h = jf['imageHeight']
    w = jf['imageWidth']

    xml_base = create_xml_base(fname, (h, w, 0))

    for k, shape in enumerate(jf['shapes']):
        contours = shape['points']

        if shape['shape_type'] == 'rectangle':
            (xmin, ymin), (xmax, ymax) = contours
        else:
            continue
            
        label = shape['label']
        xml_box = create_xml_object(xml_base, label, (xmin, ymin), (xmax, ymax))

    return xml_base


def create_xml_base(fname, img_size):
    xml_base = ET.Element(XML_ATTRIBUTE_BASE)
    xml_fname = ET.SubElement(xml_base, XML_ATTRIBUTE_FILENAME)
    xml_fname.text = fname
    
    xml_source = ET.SubElement(xml_base, XML_ATTRIBUTE_SOURCE)
    xml_database = ET.SubElement(xml_source, XML_ATTRIBUTE_DATABASE)
    xml_database.text = XML_DEFAULT_DATABASE

    h, w, c = img_size
    xml_size = ET.SubElement(xml_base, XML_ATTRIBUTE_IMGSIZE)
    xml_size_h = ET.SubElement(xml_size, XML_ATTRIBUTE_IMG_H)
    xml_size_w = ET.SubElement(xml_size, XML_ATTRIBUTE_IMG_W)
    xml_size_c = ET.SubElement(xml_size, XML_ATTRIBUTE_IMG_C)
    xml_size_h.text = str(h)
    xml_size_w.text = str(w)
    xml_size_c.text = str(c)

    xml_segmented = ET.SubElement(xml_base, XML_ATTRIBUTE_SEGMENTED)
    xml_segmented.text = XML_DEFAULT_SEGMENTED

    return xml_base


def create_xml_object(xml_base, label, pos_min, pos_max):
    xml_ojb = ET.SubElement(xml_base, XML_ATTRIBUTE_OBJECT)

    xml_name = ET.SubElement(xml_ojb, XML_ATTRIBUTE_LAVEL)
    xml_name.text = label

    xml_pase = ET.SubElement(xml_ojb, XML_ATTRIBUTE_POSE)
    xml_pase.text = XML_DEFAULT_POSE

    xml_truncate = ET.SubElement(xml_ojb, XML_ATTRIBUTE_TRUNCATED)
    xml_truncate.text = XML_DEFAULT_TRUNCATED

    xml_def = ET.SubElement(xml_ojb, XML_ATTRIBUTE_DIFFICULT)
    xml_def.text = XML_DEFAULT_DIFFICULT

    xmin, ymin = pos_min
    xmax, ymax = pos_max
    xml_box = ET.SubElement(xml_ojb, XML_ATTRIBUTE_BOX)
    xml_xmin = ET.SubElement(xml_box, XML_ATTRIBUTE_X_MIN)
    xml_xmin.text = str(xmin)
    xml_ymin = ET.SubElement(xml_box, XML_ATTRIBUTE_Y_MIN)
    xml_ymin.text = str(ymin)
    xml_xmax = ET.SubElement(xml_box, XML_ATTRIBUTE_X_MAX)
    xml_xmax.text = str(xmax)
    xml_ymax = ET.SubElement(xml_box, XML_ATTRIBUTE_Y_MAX)
    xml_ymax.text = str(ymax)

    return xml_ojb


if __name__ == '__main__':
    labelmejson_to_pascalmxl_for_dir(os.path.join('..', 'data'))
    print('all done .')

