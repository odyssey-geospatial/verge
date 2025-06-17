#
# Utilities and stuff for "geospatial entities" (GENT)
#

rules = [
    {
        "osm_key": "highway",
        "osm_values": [
            "traffic_signals"
        ],
        "gtype": "Point",
        "gent_category": "roadway feature",
        "gent_label": "traffic signals"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "crossing"
        ],
        "gtype": "Point",
        "gent_category": "roadway feature",
        "gent_label": "crosswalk"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "street_lamp"
        ],
        "gtype": "Point",
        "gent_category": "roadway feature",
        "gent_label": "street lamp"
    },
    {
        "osm_key": "highway",
        "osm_values": "footway",
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "pedestrian way"
    },
    {
        "osm_key": "highway",
        "osm_values": "bus_stop",
        "gtype": "Point",
        "gent_category": "roadway feature",
        "gent_label": "transit stop"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "motorway",
            "motorway_link"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "highway"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "primary",
            "primary_link",
            "trunk",
            "trunk_link"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "primary road"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "secondary",
            "secondary_link"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "secondary road"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "tertiary",
            "tertiary_link"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "tertiary road"
    },
    {
        "osm_key": "gtype",
        "osm_values": "LineString",
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "tertiary road"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "residential"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "residential road"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "service",
            "unclassified"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "service road"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "pedestrian",
            "steps",
            "path"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "pedestrian way"
    },
    {
        "osm_key": "highway",
        "osm_values": [
            "cycleway"
        ],
        "gtype": "LineString",
        "gent_category": "route",
        "gent_label": "cycle way"
    },
    {
        "osm_key": "waterway",
        "osm_values": [
            "river"
        ],
        "gtype": "LineString",
        "gent_category": "waterway",
        "gent_label": "river"
    },
    {
        "osm_key": "waterway",
        "osm_values": [
            "stream"
        ],
        "gtype": "LineString",
        "gent_category": "waterway",
        "gent_label": "stream"
    },
    # Polygonal rivers
    {
        "osm_key": "water",
        "osm_values": [
            "river"
        ],
        "gtype": "Polygon",
        "gent_category": "waterway",
        "gent_label": "river"
    },
    # Lakes, ponds, and reservoirs
    {
        "osm_key": "water",
        "osm_values": [
            "lake", 'pond', 'oxbow'
        ],
        "gtype": "Polygon",
        "gent_category": "waterway",
        "gent_label": "lake"
    },
    {
        "osm_key": "water",
        "osm_values": [
            'reservoir'
        ],
        "gtype": "Polygon",
        "gent_category": "waterway",
        "gent_label": "reservoir"
    },

    
    {
        "osm_key": "waterway",
        "osm_values": [
            "canal"
        ],
        "gtype": "LineString",
        "gent_category": "waterway",
        "gent_label": "canal"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "residential"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "residential"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "commercial"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "commercial"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "retail"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "retail"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "recreation_ground"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "recreation"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "cemetery"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "cemetery"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "industrial",
            "garages",
            "construction"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "industrial"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "military"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "military"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "forest"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "forest"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "meadow"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "meadow"
    },
    {
        "osm_key": "landuse",
        "osm_values": [
            "farmland",
            "orchard",
            "greenhouse_horticulture",
            "farmyard"
        ],
        "gtype": "Polygon",
        "gent_category": "landuse",
        "gent_label": "agricultural"
    },
    {
        "osm_key": "railway",
        "osm_values": [
            "rail"
        ],
        "gtype": "LineString",
        "gent_category": "railway",
        "gent_label": "rail"
    },
    {
        "osm_key": "railway",
        "osm_values": [
            "tram"
        ],
        "gtype": "LineString",
        "gent_category": "railway",
        "gent_label": "tram"
    },
    {
        "osm_key": "railway",
        "osm_values": [
            "stop",
            "station",
            "tram_stop",
            "platform"
        ],
        "gtype": "Point",
        "gent_category": "railway",
        "gent_label": "rail stop"
    },
    {
        "osm_key": "bridge",
        "osm_values": [
            "yes"
        ],
        "gtype": "LineString",
        "gent_category": "roadway feature",
        "gent_label": "bridge"
    },
    {
        "osm_key": "tunnel",
        "osm_values": [
            "yes"
        ],
        "gtype": "LineString",
        "gent_category": "roadway feature",
        "gent_label": "tunnel"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "house",
            "semidetached_house",
            "detatched",
            "house;apartments",
            "house;residential",
            "bungalow"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "house"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "apartments",
            "house;apartments",
            "yes;apartments",
            "residential",
            "barracks"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "apartment"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "commercial",
            "house;commercial",
            "retail",
            "hotel"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "commercial"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "industrial",
            "warehouse",
            "greenhouse"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "industrial"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "office",
            "civic",
            "townhall",
            "public"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "office"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "church",
            "temple"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "religious"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "garage",
            "garages",
            "parking"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "garage"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "school",
            "university"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "school"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "hospital"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "hospital"
    },
    {
        "osm_key": "building",
        "osm_values": [
            "yes"
        ],
        "gtype": "Polygon",
        "gent_category": "building",
        "gent_label": "unknown"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "parking"
        ],
        "gtype": "Polygon",
        "gent_category": "amenity",
        "gent_label": "parking lot"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "restaurant",
            "bar",
            "pub",
            "cafe",
            "coffee",
            "nightclub"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "food and drink"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "place_of_worship"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "religious"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "pitch",
            "park",
            "fountain",
            "playground",
            "outdoor_seating",
            "stadium",
            "garden"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "recreation"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "bank",
            "car_repair",
            "cannabis",
            "fitness_center",
            "alcohol",
            "mall",
            "tattoo",
            "garden_centre",
            "hairdresser",
            "clothes",
            "fast_food",
            "car",
            "books",
            "beauty",
            "florist",
            "bowling_alley",
            "swimming_pool",
            "theatre",
            "arts_centre",
            "arts",
            "veterinary",
            "appliance",
            "music",
            "events_venue",
            "erotic",
            "butcher",
            "car_wash",
            "car_rental",
            "car_parts",
            "furniture",
            "shoes",
            "fitness_centre",
            "water_park",
            "second_hand",
            "gas",
            "craft",
            "trade",
            "gift",
            "money_lender",
            "storage_rental",
            "escape_game",
            "convenience",
            "fuel",
            "travel_agency",
            "wine",
            "chocolate",
            "dance",
            "beverages",
            "confectionery"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "commercial"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "parking"
        ],
        "gtype": "Polygon",
        "gent_category": "amenity",
        "gent_label": "parking lot"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "restaurant",
            "bar",
            "pub",
            "cafe",
            "coffee",
            "nightclub"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "food and drink"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "place_of_worship"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "religious"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "pitch",
            "park",
            "fountain",
            "playground",
            "outdoor_seating",
            "stadium",
            "garden"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "recreation"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "bank",
            "car_repair",
            "cannabis",
            "fitness_center",
            "alcohol",
            "mall",
            "tattoo",
            "garden_centre",
            "hairdresser",
            "clothes",
            "fast_food",
            "car",
            "books",
            "beauty",
            "florist",
            "bowling_alley",
            "swimming_pool",
            "theatre",
            "arts_centre",
            "arts",
            "veterinary",
            "appliance",
            "music",
            "events_venue",
            "erotic",
            "butcher",
            "car_wash",
            "car_rental",
            "car_parts",
            "furniture",
            "shoes",
            "fitness_centre",
            "water_park",
            "second_hand",
            "gas",
            "craft",
            "trade",
            "gift",
            "money_lender",
            "storage_rental",
            "escape_game",
            "convenience",
            "fuel",
            "travel_agency",
            "wine",
            "chocolate",
            "dance",
            "beverages",
            "confectionery"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "commercial"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "parking"
        ],
        "gtype": "Polygon",
        "gent_category": "amenity",
        "gent_label": "parking lot"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "restaurant",
            "bar",
            "pub",
            "cafe",
            "coffee",
            "nightclub"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "food and drink"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "place_of_worship"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "religious"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "pitch",
            "park",
            "fountain",
            "playground",
            "outdoor_seating",
            "stadium",
            "garden"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "recreation"
    },
    {
        "osm_key": "amenity",
        "osm_values": [
            "bank",
            "car_repair",
            "cannabis",
            "fitness_center",
            "alcohol",
            "mall",
            "tattoo",
            "garden_centre",
            "hairdresser",
            "clothes",
            "fast_food",
            "car",
            "books",
            "beauty",
            "florist",
            "bowling_alley",
            "swimming_pool",
            "theatre",
            "arts_centre",
            "arts",
            "veterinary",
            "appliance",
            "music",
            "events_venue",
            "erotic",
            "butcher",
            "car_wash",
            "car_rental",
            "car_parts",
            "furniture",
            "shoes",
            "fitness_centre",
            "water_park",
            "second_hand",
            "gas",
            "craft",
            "trade",
            "gift",
            "money_lender",
            "storage_rental",
            "escape_game",
            "convenience",
            "fuel",
            "travel_agency",
            "wine",
            "chocolate",
            "dance",
            "beverages",
            "confectionery"
        ],
        "gtype": "Point",
        "gent_category": "amenity",
        "gent_label": "commercial"
    }
]