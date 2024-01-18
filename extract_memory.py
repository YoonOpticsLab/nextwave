import numpy as np

numbytes={
    'uint8_t':1,
    'uint16_t':2,
    'double':8,
    'float':4,
    'int64_t':8, # longlong
    'uint64_t':8 # longlong
}

struct_codes={
    'uint8_t':'B',
    'uint16_t':'H',
    'double':'d',
    'float':'f',
    'int64_t':'q', # longlong
    'uint64_t':'Q' # longlong
} 

def pytpe(str_type):
    if str_type in ['float','double']:
        return str_type
    else:
        return str_type[:-2]

# https://docs.python.org/3.9/library/struct.html

def get_item(layout,bytes,which_item):
    entry = layout[1][which_item]
    loc=entry['bytenum_current']
    itemsize=numbytes[ entry['type'] ]
    print(entry['type'], loc, itemsize)
    data = np.frombuffer(bytes[loc:loc+itemsize], pytpe(entry['type']), count=1)[0]
    return data

def get_array_item(layout,bytes,which_item, which_index):
    entry = layout[1][which_item]
    array_start=entry['bytenum_current']

    itemsize=numbytes[ entry['type'] ]
    first=array_start+itemsize*which_index
    data = np.frombuffer(bytes[first:first+itemsize], pytpe(entry['type']), count=1)[0]
    if 'int' in pytpe(entry['type']):
        data=int(data) # Make an untyped int size (in case we are in something like a multiply
    return data

def get_header_format(filname):
    with open(filname) as f:
        lines=f.readlines()

    defs={}    # in structure
    fields={}  # in structure
    nfield=0
    bytenum_current=0

    for lin1 in lines:

        # Replace any preprocessor #defines
        for key1 in defs.keys():
            lin1=lin1.replace(key1,defs[key1])

        fields1=lin1.strip().split(' ') # just this line

        # Skip special lines (defines, brackets, struct line)
        if lin1.find('#define')>=0:
            defs[fields1[1]]=fields1[2]
            continue
        elif lin1.find('struct')>=0:
            continue
        elif len(fields1)<2:
            continue

        var_name=fields1[1].strip(';')
        var_type=fields1[0]
        num_elements=1
        var_init=""

        if var_name.find('[')>=0:
            name_parse=var_name.split('[')
            var_name=name_parse[0]
            num_elements=int(name_parse[1][:-1])
        elif var_name.find('=')>=0:
            name_parse=var_name.split('=')
            var_name=name_parse[0]
            var_init=name_parse[1]

        num_bytes1=numbytes[var_type]*num_elements

       # print(var_type)
        fields[var_name]={'type':fields1[0],'num_elements':num_elements,
                          'num_bytes':num_bytes1,'bytenum_current':bytenum_current,
                          'init':var_init}
        bytenum_current += num_bytes1

        #print( var_name )
    header_size=bytenum_current
    return (header_size,fields,defs)
