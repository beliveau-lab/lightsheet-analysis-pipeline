import xml.etree.ElementTree as ET
import re
import argparse
import logging

logger = logging.getLogger(__name__)
def modify_spimdata_xml(input_filepath, output_filepath):
    """
    Modifies a SpimData XML file to perform a 90 degree rotation of the sample.
    """
    try:
        tree = ET.parse(input_filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {input_filepath}: {e}")
        return
    except FileNotFoundError:
        print(f"Error: Input file {input_filepath} not found.")
        return

    # 1. Modify BasePath
    base_path_element = root.find('BasePath')
    if base_path_element is not None:
        base_path_element.set('type', 'relative')
        base_path_element.text = '.'
    else:
        # If BasePath somehow doesn't exist, create it (optional, depends on strictness)
        # For this problem, we assume it exists as per input.xml
        print("Warning: BasePath element not found.")


    # 2. Modify ViewSetups
    sequence_description = root.find('SequenceDescription')
    if sequence_description is None:
        print("Warning: SequenceDescription element not found.")
        return

    view_setups_parent = sequence_description.find('ViewSetups')
    if view_setups_parent is None:
        print("Warning: ViewSetups element not found.")
        return

    # 2a. Reorder gridLocation within each ViewSetup
    for vs_elem in view_setups_parent.findall('ViewSetup'):
        id_elem = vs_elem.find('id')
        grid_location_elem = vs_elem.find('gridLocation')

        if id_elem is not None and grid_location_elem is not None:
            # Remove gridLocation from its current position
            vs_elem.remove(grid_location_elem)
            # Find the index of id_elem and insert gridLocation after it
            children = list(vs_elem)
            id_idx = -1
            for i, child in enumerate(children):
                if child == id_elem:
                    id_idx = i
                    break
            if id_idx != -1:
                vs_elem.insert(id_idx + 1, grid_location_elem)
            else: # Should not happen if id_elem was found
                vs_elem.append(grid_location_elem) # Fallback: append if id somehow not in children list

    # 2b. Remove Attributes name="gridShape"
    grid_shape_attrs_to_remove = None
    for attrs_elem in view_setups_parent.findall('Attributes'):
        if attrs_elem.get('name') == 'gridShape':
            grid_shape_attrs_to_remove = attrs_elem
            break
    if grid_shape_attrs_to_remove is not None:
        view_setups_parent.remove(grid_shape_attrs_to_remove)

    # 2c. Add Attributes name="angle"
    # Create the new Attributes element for "angle"
    angle_attributes = ET.Element('Attributes', {'name': 'angle'})

    # Create and append Illumination sub-element
    illum_angle = ET.SubElement(angle_attributes, 'Illumination')
    ET.SubElement(illum_angle, 'id').text = '0'
    ET.SubElement(illum_angle, 'name').text = '0'

    # Create and append Angle sub-element
    angle_angle = ET.SubElement(angle_attributes, 'Angle')
    ET.SubElement(angle_angle, 'id').text = '0'
    ET.SubElement(angle_angle, 'name').text = '0'

    # Find the position to insert the new "angle" Attributes.
    # It should go after the last existing "Attributes" tag (e.g., "tile").
    insert_after_idx = -1
    children_of_viewsetups = list(view_setups_parent)
    for i, child in enumerate(children_of_viewsetups):
        if child.tag == 'Attributes': # Find the last one
            insert_after_idx = i

    if insert_after_idx != -1:
        view_setups_parent.insert(insert_after_idx + 1, angle_attributes)
    else:
        # If no other Attributes elements (unlikely for valid input),
        # find ViewSetup elements and insert after the last one.
        last_vs_idx = -1
        for i, child in enumerate(children_of_viewsetups):
            if child.tag == 'ViewSetup':
                last_vs_idx = i
        if last_vs_idx != -1:
            view_setups_parent.insert(last_vs_idx + 1, angle_attributes)
        else: # Fallback: just append
            view_setups_parent.append(angle_attributes)


    # 3. Modify ViewRegistrations
    view_registrations_parent = root.find('ViewRegistrations')
    if view_registrations_parent is not None:
        affine_text_val = "1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0"
        transform_name_val = "Manually defined transformation (Rigid transform defined by BigDataViewer)"

        for vr_elem in view_registrations_parent.findall('ViewRegistration'):
            # Create the new ViewTransform element
            new_vt = ET.Element('ViewTransform', {'type': 'affine'})
            ET.SubElement(new_vt, 'Name').text = transform_name_val
            ET.SubElement(new_vt, 'affine').text = affine_text_val
            # Insert it as the first child
            vr_elem.insert(0, new_vt)
    else:
        print("Warning: ViewRegistrations element not found.")

    # 4. Write the modified XML
    # ET.indent is available in Python 3.9+ for pretty printing
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ") # Indent with 2 spaces
    else:
        # Basic pretty print for older versions (might not be perfect)
        # For more control, you might use lxml or a custom function
        def pretty_print(current, parent=None, index=-1, depth=0):
            for i, node in enumerate(current):
                pretty_print(node, current, i, depth + 1)
            if parent is not None:
                if index == 0:
                    parent.text = '\n' + ('  ' * depth)
                else:
                    parent[index - 1].tail = '\n' + ('  ' * depth)
                if index == len(parent) - 1:
                    current.tail = '\n' + ('  ' * (depth - 1))
        # pretty_print(root) # This basic pretty printer is often not robust enough.
        # ET.indent is preferred. If not available, output might not be perfectly formatted.
        pass


    try:
        tree.write(output_filepath, encoding='UTF-8', xml_declaration=True)
        print(f"Successfully modified XML and saved to {output_filepath}")
    except IOError as e:
        print(f"Error writing XML to file {output_filepath}: {e}")

if __name__ == "__main__":

    # --- Argument Parsing ---
    # Check if running under Snakemake
    if 'snakemake' in globals():
        logger.info("Running under Snakemake, using snakemake object for parameters.")
        args = argparse.Namespace(
            input_xml=snakemake.input.xml,
            output_xml=snakemake.output.xml,
        )
    else:
        logger.info("Not running under Snakemake, parsing command-line arguments.")
        args = argparse.Namespace(
            input_xml=snakemake.input.xml,
            output_xml=snakemake.output.xml,
        )

    modify_spimdata_xml(args.input_xml, args.output_xml)