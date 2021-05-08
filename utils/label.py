from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'color'         # The color of this label (RGB)
    ] )

labels = [
    #       name                     id  category           catId           color
    Label(  'unlabeled'            ,  0 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'static'               ,  4 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 , 'void'            , 0       , (111, 74,  0) ),
    Label(  'ground'               ,  6 , 'void'            , 0       , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 , 'flat'            , 1       , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 , 'flat'            , 1       , (244, 35,232) ),
    Label(  'parking'              ,  9 , 'flat'            , 1       , (250,170,160) ),
    Label(  'rail track'           , 10 , 'flat'            , 1       , (230,150,140) ),
    Label(  'building'             , 11 , 'construction'    , 2       , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 , 'construction'    , 2       , (102,102,156) ),
    Label(  'fence'                , 13 , 'construction'    , 2       , (190,153,153) ),
    Label(  'guard rail'           , 14 , 'construction'    , 2       , (180,165,180) ),
    Label(  'bridge'               , 15 , 'construction'    , 2       , (150,100,100) ),
    Label(  'tunnel'               , 16 , 'construction'    , 2       , (150,120, 90) ),
    Label(  'pole'                 , 17 , 'object'          , 3       , (153,153,153) ),
    Label(  'polegroup'            , 18 , 'object'          , 3       , (153,153,153) ),
    Label(  'traffic light'        , 19 , 'object'          , 3       , (250,170, 30) ),
    Label(  'traffic sign'         , 20 , 'object'          , 3       , (220,220,  0) ),
    Label(  'vegetation'           , 21 , 'nature'          , 4       , (107,142, 35) ),
    Label(  'terrain'              , 22 , 'nature'          , 4       , (152,251,152) ),
    Label(  'sky'                  , 23 , 'sky'             , 5       , ( 70,130,180) ),
    Label(  'person'               , 24 , 'human'           , 6       , (220, 20, 60) ),
    Label(  'rider'                , 25 , 'human'           , 6       , (255,  0,  0) ),
    Label(  'car'                  , 26 , 'vehicle'         , 7       , (  0,  0,142) ),
    Label(  'truck'                , 27 , 'vehicle'         , 7       , (  0,  0, 70) ),
    Label(  'bus'                  , 28 , 'vehicle'         , 7       , (  0, 60,100) ),
    Label(  'caravan'              , 29 , 'vehicle'         , 7       , (  0,  0, 90) ),
    Label(  'trailer'              , 30 , 'vehicle'         , 7       , (  0,  0,110) ),
    Label(  'train'                , 31 , 'vehicle'         , 7       , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 , 'vehicle'         , 7       , (  0,  0,230) ),
    Label(  'bicycle'              , 33 , 'vehicle'         , 7       , (119, 11, 32) ),
    Label(  'license plate'        , -1 , 'vehicle'         , 7       , (  0,  0,142) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }

# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>14} | {:>10}".format( 'name', 'id', 'category', 'categoryId'))
    print("    " + ('-' * 98))
    for label in labels:
        print(" \"{:}\"".format(label.name))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))
