# tool_registry.py

TOOLS = {
    "get_flight_basic_info": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Fetch basic flight information including carrier, flight number, stations, scheduled times, and flight status.",
    },
    "get_equipment_info": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Get aircraft equipment details: aircraft type, tail number (registration), and configuration.",
    },
    "get_operation_times": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Return estimated and actual operation times: takeoff, landing, departure, arrival, and block times.",
    },
    "get_fuel_summary": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Retrieve fuel summary including planned vs actual fuel consumption for the flight.",
    },
    "get_delay_summary": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Get delay information including delay reasons, durations, and total delay time.",
    },
    "health_check": {
        "args": [],
        "desc": "Check the health status of the MCP server and database connection.",
    },
    "raw_mongodb_query": {
        "args": ["query_json", "limit"],
        "desc": "Run a raw MongoDB query (JSON format) for debugging purposes.",
    },
}