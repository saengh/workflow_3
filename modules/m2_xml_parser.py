from m1_main import *

import pandas as pd
from lxml import etree

def parse_xml(input_xml_path, input_xml_fields):

    data_list = []  # List to hold data for each QOdocument
    context = etree.iterparse(input_xml_path, events=('end',), tag='QOdocument')
    
    for event, element in context:
        # current_data = {'QN': element.attrib.get('QN', ''), 'PNKC': ''}  # Initialize data for the current QOdocument
        current_data = {'PNKC': ''}  # Initialize data for the current QOdocument
        
        # Loop through QOfield within QOdocument
        for qofield in element.findall('.//QOfield'):
            field_name = qofield.attrib.get('name')
            
            if field_name == 'PN':
               for qopar in qofield.findall('.//QOpar'):
                  current_data['PNKC'] = qopar.attrib.get('PUB', '')
                  break

            if field_name in input_xml_fields and field_name not in current_data:
               # Initialize field in current_data if not present
               current_data[field_name] = ""

               # Aggregate text for the current field
               for qopar in qofield.findall('.//QOpar'):
                  texts = ' '.join(qopar.itertext())
                  current_data[field_name] += texts + ' '  # Add space for separation

        data_list.append(current_data)  # Add the current document's data to the list
        clear_element(element)  # Clear processed element from memory

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df

def clear_element(element):
    """
    Clear the given element from memory to optimize memory usage during parsing.
    :param element: The element to clear from memory.
    """
    element.clear()
    # Also remove the element from its parent to further reduce memory usage
    while element.getprevious() is not None:
        del element.getparent()[0]

def delimit_pnkc(pnkc):
    if pnkc.count('-') >= 2:
        return pnkc.rsplit('-', maxsplit=1)[0]
    else:
        return pnkc
    
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

df = parse_xml(input_xml_path, input_xml_fields)

# Create a combined fields (CTB) column
df['CTB'] = df[['ETI', 'EAB', 'ECLM']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Delimit PNKC from right and remove hyphen after country code
df['PNKC'] = df['PNKC'].apply(delimit_pnkc)
df['PNKC'] = df['PNKC'].str.replace('-', '', n=1)

# Rearrange df columns
df = df[['PNKC', 'ETI', 'EAB', 'ICLM', 'ECLM', 'CTB', 'CPC']]

# Renaming columns
new_column_names = ['PNKC', 'TI', 'AB', 'ICLM', 'CLMS', 'CTB', 'CPC']
df.columns = new_column_names

# Saving dataframe
df.to_excel(workflow_folder + r'\excel\parsed_xml.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\parsed_xml.pickle')