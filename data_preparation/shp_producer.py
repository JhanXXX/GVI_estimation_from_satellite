def createPoints(inshp, outshp, mini_dist):
    '''
    This function will parse through the street network of provided city and
    clean all highways and create points every mini_dist meters (or as specified) along
    the linestrings
    Required modules: Fiona and Shapely

    parameters:
        inshp: the input linear shapefile, must be in WGS84 projection, ESPG: 4326
        output: the result point feature class
        mini_dist: the minimum distance between two created point

    last modified by Xiaojiang Li, MIT Senseable City Lab
    '''
    
    import fiona
    import os, os.path
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    from functools import partial
    import pyproj
    from fiona.crs import from_epsg
    
    count = 0
    s = {'trunk_link','tertiary','motorway','motorway_link','steps', None, ' ','pedestrian','primary', 'primary_link','footway','tertiary_link', 'trunk','secondary','secondary_link','tertiary_link','bridleway','service'}
    
    # the temporary file of the cleaned data
    root = os.path.dirname(inshp)
    basename = 'clean_' + os.path.basename(inshp)
    temp_cleanedStreetmap = os.path.join(root, basename)
    
    # if the tempfile exist then delete it
    if os.path.exists(temp_cleanedStreetmap):
        fiona.remove(temp_cleanedStreetmap, 'ESRI Shapefile')
    
    # clean the original street maps by removing highways
    with fiona.open(inshp) as source, fiona.open(temp_cleanedStreetmap, 'w', driver=source.driver, crs=source.crs, schema=source.schema) as dest:
        
        for feat in source:
            try:
                i = feat['properties']['highway'] # for the OSM street data
                if i in s:
                    continue
            except (KeyError, TypeError):
                # 修复：正确获取第一个属性字段名
                try:
                    # 将 keys() 转换为列表再访问
                    property_keys = list(dest.schema['properties'].keys())
                    if property_keys:  # 确保有属性字段
                        key = property_keys[0]
                        i = feat['properties'][key]
                        if i in s:
                            continue
                except (IndexError, KeyError, TypeError):
                    # 如果获取属性失败，跳过这个要素
                    pass
            
            dest.write(feat)

    schema = {
        'geometry': 'Point',
        'properties': {'id': 'int'},
    }

    # Create points along the streets
    with fiona.drivers():
        with fiona.open(outshp, 'w', crs=from_epsg(4326), driver='ESRI Shapefile', schema=schema) as output:
            for line in fiona.open(temp_cleanedStreetmap):
                first = shape(line['geometry'])
                
                length = first.length
                
                try:
                    # 修复：使用现代 pyproj 语法
                    # 创建坐标转换器：WGS84 -> Web Mercator
                    transformer_to_meter = pyproj.Transformer.from_crs(
                        'EPSG:4326', 'EPSG:3857', always_xy=True
                    )
                    
                    # 转换到米制坐标系
                    line2 = transform(transformer_to_meter.transform, first)
                    linestr = list(line2.coords)
                    dist = mini_dist
                    
                    for distance in range(0, int(line2.length), dist):
                        point = line2.interpolate(distance)
                        
                        # 创建反向转换器：Web Mercator -> WGS84
                        transformer_to_wgs84 = pyproj.Transformer.from_crs(
                            'EPSG:3857', 'EPSG:4326', always_xy=True
                        )
                        
                        # WGS84
                        point = transform(transformer_to_wgs84.transform, point)
                        output.write({'geometry': mapping(point), 'properties': {'id': 1}})
                        
                except Exception as e:
                    print(f"Error processing line: {e}")
                    print("Make sure the input shapefile is in WGS84 projection")
                    continue
                    
    print("Process Complete")
    
    # delete the temporary cleaned shapefile
    try:
        fiona.remove(temp_cleanedStreetmap, 'ESRI Shapefile')
    except:
        print(f"Warning: Could not remove temporary file {temp_cleanedStreetmap}")


# Example to use the code, 
# Note: make sure the input linear featureclass (shapefile) is in WGS 84 or ESPG: 4326
# ------------main ----------
if __name__ == "__main__":
    import os,os.path
    import sys

    """
    collections = {
        "Sweden":["Stockholm"],
        "Germany": ["Berlin","Hamburg"],
        "Spain": ["Barcelona"],
        "Denmark":["Copenhagen"],
        "UK":["London"],
        "Greece":["Athens"],
        "France":["Paris"],
        "Netherlands":["Amsterdam"],
        "Hungary":["Budapest"]
    }

    collections = {
        "Italy":["Bologna","Milan"],
        "Estonia":["Tallinn"],
        "Sweden":["Gothenburg"]
    }
    """

    collections = {
    "UK":["Manchester"],
    "Germany":["Dusseldorf","Cologne"],
    "Switzerland":["Zurich"]
    }


    mini_dist = 50 #the minimum distance between two generated points in meter

    inshp = "/workspace/data/resource/Sweden/Stockholm_jr_original_2.shp"
    outshp = "/workspace/data/resource/Sweden/Stockholm_jr_samples_2.shp"
    createPoints(inshp, outshp, mini_dist)

    """for country in collections.keys():
        cities = collections[country]
        for city in cities:
            print(f"Processing {city}, {country}...")
            root = f'./data/resource/{country}'
            inshp = os.path.join(root, f'{city}_original.shp')
            outshp = os.path.join(root, f'{city}.shp')
            createPoints(inshp, outshp, mini_dist)"""