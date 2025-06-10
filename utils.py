#
# Utilities for use with the "blip" project.
#


    {
        'key': 'highway',
        'values': ['crossing'],
        'gtype': 'Point',
        'category': 'roadway feature',
        'label': 'traffic signals',
    },



highway_rules = [
    {
        'category': 'roadway feature',
        'label': 'traffic signals',
        'gtype': 'Point',
        'keys': {'highway': ['crossing'], 'crossing': ['traffic_signals']},
    },
    {
        'category': 'roadway feature',
        'label': 'traffic signals',
        'gtype': 'Point',
        'keys': {'highway': ['traffic_signals']},
    },
    {
        'category': 'roadway feature',
        'label': 'crosswalk', 
        'gtype': 'Point',
        'keys': {'highway': ['crossing'], 'crossing': ['marked']}
    },
    {
        'category': 'roadway feature',
        'label': 'crosswalk', 
        'gtype': 'Point',
        'keys': {'highway': ['crossing']}
    },
    {
        'category': 'roadway feature',
        'label': 'street lamp',
        'gtype': 'Point',
        'keys': {'highway': ['street_lamp']}
    },
    {
        'category': 'route',
        'label': 'pedestrian way', 
        'gtype': 'LineString',
        'keys': {'highway': 'footway'}
    },
    {
        'category': 'roadway feature',
        'label': 'transit stop', 
        'gtype': 'Point',
        'keys': {'highway': 'bus_stop'}
    },
    {
        'category': 'route',
        'label': 'highway', 
        'gtype': 'LineString',
        'keys': {'highway': ['motorway', 'motorway_link']}
    },
    {
        'category': 'route',
        'label': 'primary road', 
        'gtype': 'LineString',
        'keys': {'highway': ['primary', 'primary_link', 'trunk', 'trunk_link']}
    },
    {
        'category': 'route',
        'label': 'secondary road',
        'gtype': 'LineString',
        'keys': {'highway': ['secondary', 'secondary_link']}
    },
    {
        'category': 'route',
        'label': 'tertiary road', 
        'gtype': 'LineString',
        'keys': {'highway': ['tertiary', 'tertiary_link'], 'gtype': 'LineString'}
    },
    {
        'category': 'route',
        'label': 'residential road', 
        'gtype': 'LineString',
        'keys': {'highway': ['residential']}
    },
    {
        'category': 'route',
        'label': 'service road', 
        'gtype': 'LineString',
        'keys': {'highway': ['service', 'unclassified']}
    },
    {
        'category': 'route',
        'label': 'pedestrian way', 
        'gtype': 'LineString',
        'keys': {'highway': ['pedestrian', 'steps', 'path']}
    },
    {
        'category': 'route',
        'label': 'cycle way', 
        'gtype': 'LineString',
        'keys': {'highway': ['cycleway']}
    },
]


waterway_rules = [
    {
        'category': 'waterway',
        'label': 'river',
        'gtype': 'LineString',
        'keys': {'waterway': ['river']},
    },
    {
        'category': 'waterway',
        'label': 'stream',
        'gtype': 'LineString',
        'keys': {'waterway': ['stream']},
    },
    {
        'category': 'waterway',
        'label': 'canal',
        'gtype': 'LineString',
        'keys': {'waterway': ['canal']},
    },
]


landuse_rules = [
    {
        'category': 'landuse',
        'label': 'residential',
        'gtype': 'Polygon',
        'keys': {'landuse': ['residential']},
    },
    {
        'category': 'landuse',
        'label': 'commercial',
        'gtype': 'Polygon',
        'keys': {'landuse': ['commercial']},
    },
    {
        'category': 'landuse',
        'label': 'retail',
        'gtype': 'Polygon',
        'keys': {'landuse': ['retail']},
    },
    {
        'category': 'landuse',
        'label': 'recreation',
        'gtype': 'Polygon',
        'keys': {'landuse': ['recreation_ground']},
    },
    {
        'category': 'landuse',
        'label': 'cemetery',
        'gtype': 'Polygon',
        'keys': {'landuse': ['cemetery']},
    },
    {
        'category': 'landuse',
        'label': 'industrial',
        'gtype': 'Polygon',
        'keys': {'landuse': ['industrial', 'garages', 'construction']},
    },
    {
        'category': 'landuse',
        'label': 'military',
        'gtype': 'Polygon',
        'keys': {'landuse': ['military']},
    },
    {
        'category': 'landuse',
        'label': 'forest',
        'gtype': 'Polygon',
        'keys': {'landuse': ['forest']},
    },
    {
        'category': 'landuse',
        'label': 'meadow',
        'gtype': 'Polygon',
        'keys': {'landuse': ['meadow']},
    },
    {
        'category': 'landuse',
        'label': 'agricultural',
        'gtype': 'Polygon',
        'keys': {'landuse': ['farmland', 'orchard', 'greenhouse_horticulture', 'farmyard']},
    },
]

railway_rules = [
    {
        'category': 'railway',
        'label': 'rail',
        'gtype': 'LineString',
        'keys': {'railway': ['rail']},
    },
    {
        'category': 'railway',
        'label': 'tram',
        'gtype': 'LineString',
        'keys': {'railway': ['tram']},
    },
    {
        'category': 'railway',
        'label': 'rail stop',
        'gtype': 'Point',
        'keys': {'railway': ['stop', 'station', 'tram_stop', 'platform']},
    },
]

bridge_rules = [
    {
        'category': 'bridge',
        'label': 'bridge',
        'gtype': 'LineString',
        'keys': {'bridge': ['yes']},
    },
]

tunnel_rules = [
    {
        'category': 'tunnel',
        'label': 'tunnel',
        'gtype': 'LineString',
        'keys': {'tunnel': ['yes']},
    },
]

building_rules = [
    {
        'category': 'building',
        'label': 'house',
        'gtype': 'Polygon',
        'keys': {'building': ['house', 'semidetached_house', 'detatched', 'house;apartments', 'house;residential',
                             'bungalow']},
    },
    {
        'category': 'building',
        'label': 'apartment',
        'gtype': 'Polygon',
        'keys': {'building': ['apartments', 'house;apartments', 'yes;apartments', 'residential', 'barracks']},
    },
    {
        'category': 'building',
        'label': 'commercial',
        'gtype': 'Polygon',
        'keys': {'building': ['commercial', 'house;commercial', 'retail', 'hotel', ]},
    },
    {
        'category': 'building',
        'label': 'industrial',
        'gtype': 'Polygon',
        'keys': {'building': ['industrial', 'warehouse', 'greenhouse']},
    },
    {
        'category': 'building',
        'label': 'office',
        'gtype': 'Polygon',
        'keys': {'building': ['office', 'civic', 'townhall', 'public']},
    },
    {
        'category': 'building',
        'label': 'religious',
        'gtype': 'Polygon',
        'keys': {'building': ['church', 'temple', ]},
    },
    {
        'category': 'building',
        'label': 'garage',
        'gtype': 'Polygon',
        'keys': {'building': ['garage', 'garages', 'parking']},
    },
    {
        'category': 'building',
        'label': 'school',
        'gtype': 'Polygon',
        'keys': {'building': ['school', 'university']},
    },
    {
        'category': 'building',
        'label': 'hospital',
        'gtype': 'Polygon',
        'keys': {'building': ['hospital']},
    },
    {
        'category': 'building',
        'label': 'unknown',
        'gtype': 'Polygon',
        'keys': {'building': ['yes', ]},
    },
]

amenity_rules = [
    {
        'category': 'amenity',
        'label': 'parking lot',
        'gtype': 'Polygon',
        'keys': {'amenity': ['parking']},
    },
    {
        'category': 'amenity',
        'label': 'food and drink',
        'gtype': 'Point',
        'keys': {'amenity': ['restaurant', 'bar', 'pub', 'cafe', 'coffee', 'nightclub']},
    },
    {
        'category': 'amenity',
        'label': 'religious',
        'gtype': 'Point',
        'keys': {'amenity': ['place_of_worship']},
    },
    {
        'category': 'amenity',
        'label': 'recreation',
        'gtype': 'Point',
        'keys': {'amenity': ['pitch', 'park', 'fountain', 'playground', 'outdoor_seating', 'stadium', 'garden', ]},
    },
    {
        'category': 'amenity',
        'label': 'commercial',
        'gtype': 'Point',
        'keys': {'amenity': [
            'bank', 'car_repair', 'cannabis', 'fitness_center', 'alcohol', 'mall', 'tattoo', 'garden_centre',
            'hairdresser', 'clothes', 'fast_food', 'car', 'books', 'beauty', 'florist', 
            'bowling_alley', 'swimming_pool', 'theatre', 'arts_centre', 'arts', 'veterinary', 'appliance',
            'music', 'events_venue', 'erotic', 'butcher', 'car_wash', 'car_rental', 'car_parts', 'furniture',
            'shoes', 'fitness_centre', 'water_park', 'second_hand', 'gas', 'craft', 'trade', 'gift',
            'money_lender', 'storage_rental', 'escape_game', 'convenience', 'fuel', 'travel_agency',
            'wine', 'chocolate', 'dance', 'beverages', 'confectionery'
        ]},
    },
]


rules = {
    'highway': highway_rules,
    'waterway': waterway_rules,
    'landuse': landuse_rules,
    'railway': railway_rules,
    'bridge': bridge_rules,
    'tunnel': tunnel_rules,
    'building': building_rules,
    'amenity': amenity_rules,
    'shop': amenity_rules,  # yes, that is correct
    'leisure': amenity_rules,  # yes, that is correct
    
}

