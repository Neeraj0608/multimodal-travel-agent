"""Curated corpus for the three cities the agent knows "from memory".

Each city is split into several topical chunks rather than one blob, so
retrieval can return the parts that actually match the question and the routing
decision has more than one signal to work with.

Anything not in here (Kyoto, Snohomish, ...) falls through to the web-search
path instead.
"""

from __future__ import annotations

from typing import TypedDict


class CityDoc(TypedDict):
    id: str
    city: str
    country: str
    section: str
    text: str


CITY_DOCUMENTS: list[CityDoc] = [
    # ---------------------------------------------------------------- Paris
    {
        "id": "paris-overview",
        "city": "Paris",
        "country": "France",
        "section": "overview",
        "text": (
            "Paris, the capital of France, sits on the Seine in the north of the country and "
            "is organised as twenty arrondissements spiralling clockwise from the Louvre. "
            "The city is compact by global-capital standards, roughly 105 square kilometres, "
            "which makes it unusually walkable: most visitors cross from the Marais to "
            "Saint-Germain on foot in under forty minutes. Its identity was shaped by "
            "Haussmann's 19th-century rebuilding, which produced the wide boulevards, "
            "uniform cream limestone facades and zinc rooftops that define the skyline."
        ),
    },
    {
        "id": "paris-attractions",
        "city": "Paris",
        "country": "France",
        "section": "attractions",
        "text": (
            "The Louvre holds roughly 35,000 works on display and rewards a targeted visit "
            "rather than an exhaustive one. The Musee d'Orsay, in a converted Beaux-Arts "
            "railway station, has the strongest Impressionist collection in the world. "
            "Sainte-Chapelle's upper chapel is fifteen metres of 13th-century stained glass "
            "and is best seen on a bright morning. Notre-Dame reopened to visitors after the "
            "2019 fire restoration. The Eiffel Tower is busiest at sunset; the Trocadero "
            "terrace opposite gives the classic view without the queue."
        ),
    },
    {
        "id": "paris-food",
        "city": "Paris",
        "country": "France",
        "section": "food",
        "text": (
            "Parisian eating runs on rhythm: bakery in the morning, a long lunch formule "
            "around 12:30, dinner rarely before 20:00. Look for the 'boulangerie artisanale' "
            "label, which signals bread baked on the premises. The bistro revival is "
            "concentrated in the 11th arrondissement around Rue de Charonne, while the 20th "
            "has become the city's most interesting low-key food neighbourhood. Marche "
            "d'Aligre is the best everyday market: covered hall, outdoor stalls and a "
            "flea market in one block."
        ),
    },
    {
        "id": "paris-transport",
        "city": "Paris",
        "country": "France",
        "section": "transport",
        "text": (
            "The Metro has sixteen lines and a station within 500 metres of almost anywhere "
            "inside the peripherique. Navigo Easy is the practical tourist card. RER B links "
            "Charles de Gaulle airport to the centre in about 35 minutes and is usually "
            "faster than a taxi. Velib bike share covers the city and the riverside "
            "expressway on the Right Bank is now a pedestrian promenade."
        ),
    },
    {
        "id": "paris-when",
        "city": "Paris",
        "country": "France",
        "section": "best time to visit",
        "text": (
            "Late April to June and September to early October are the strongest windows: "
            "mild temperatures, long daylight and gardens in full use. August empties out as "
            "Parisians leave, so some family-run restaurants close, but queues shorten and "
            "Paris Plages turns the riverbank into a beach. Winter is grey and damp but "
            "museum crowds thin dramatically."
        ),
    },
    # ---------------------------------------------------------------- Tokyo
    {
        "id": "tokyo-overview",
        "city": "Tokyo",
        "country": "Japan",
        "section": "overview",
        "text": (
            "Tokyo is less a single city than a federation of twenty-three special wards, "
            "each with its own centre of gravity. The Greater Tokyo Area holds around 37 "
            "million people, making it the largest metropolitan region on earth, yet it runs "
            "quietly and punctually. Orientation is easiest by rail node: Shinjuku for "
            "nightlife and skyscrapers, Shibuya for youth culture, Ueno for museums and "
            "parks, Asakusa for the older low city, Marunouchi for corporate Tokyo."
        ),
    },
    {
        "id": "tokyo-attractions",
        "city": "Tokyo",
        "country": "Japan",
        "section": "attractions",
        "text": (
            "Senso-ji in Asakusa is the city's oldest temple, founded in 645, approached "
            "through the Nakamise shopping street. The Meiji Jingu shrine sits inside a "
            "planted forest of 100,000 donated trees beside Harajuku. teamLab's digital art "
            "museums require advance timed tickets. The Tokyo National Museum in Ueno holds "
            "the deepest collection of Japanese art anywhere. For views, the Tokyo "
            "Metropolitan Government Building observatory in Shinjuku is free."
        ),
    },
    {
        "id": "tokyo-food",
        "city": "Tokyo",
        "country": "Japan",
        "section": "food",
        "text": (
            "Tokyo holds more Michelin stars than any other city, but its real strength is "
            "the specialist counter: a shop that makes only tempura, only soba, or only "
            "tonkatsu, often for generations. Department store basements, called depachika, "
            "are the fastest way to sample high-quality prepared food. Toyosu Market "
            "replaced Tsukiji for the tuna auction, while Tsukiji's outer market still "
            "trades in knives, dried goods and breakfast sushi. Standing bars under the "
            "Yamanote line tracks in Yurakucho are cheap and lively after work."
        ),
    },
    {
        "id": "tokyo-transport",
        "city": "Tokyo",
        "country": "Japan",
        "section": "transport",
        "text": (
            "Rail is the default: the JR Yamanote loop connects most major districts and the "
            "Tokyo Metro fills in the middle. A Suica or Pasmo IC card works across every "
            "operator plus convenience stores. Trains stop around 00:30 and restart near "
            "05:00, and taxis after midnight are expensive. Narita is 60 to 90 minutes out; "
            "Haneda is 30 minutes and far more convenient."
        ),
    },
    {
        "id": "tokyo-when",
        "city": "Tokyo",
        "country": "Japan",
        "section": "best time to visit",
        "text": (
            "Late March to early April brings cherry blossom and the year's heaviest "
            "domestic travel. Late October to November is the quieter equal: clear skies, "
            "cool air and autumn colour in the gardens. Avoid the rainy season in June and "
            "the humid heat of August. Golden Week in early May books out nationwide."
        ),
    },
    # ------------------------------------------------------------- New York
    {
        "id": "newyork-overview",
        "city": "New York",
        "country": "United States",
        "section": "overview",
        "text": (
            "New York City spans five boroughs - Manhattan, Brooklyn, Queens, the Bronx and "
            "Staten Island - and about 8.3 million residents speaking more than 200 "
            "languages. Manhattan's street grid above 14th Street makes navigation trivial: "
            "avenues run north-south, streets east-west, and twenty blocks make a mile. "
            "Downtown below 14th keeps its older tangled street pattern, which is why the "
            "Village feels different to walk than Midtown."
        ),
    },
    {
        "id": "newyork-attractions",
        "city": "New York",
        "country": "United States",
        "section": "attractions",
        "text": (
            "Central Park covers 843 acres and is entirely designed landscape, not preserved "
            "wilderness. The Metropolitan Museum of Art spans 5,000 years across two million "
            "objects. MoMA holds the modern canon; the Whitney anchors the High Line's "
            "southern end. The 9/11 Memorial's twin voids sit in the footprints of the "
            "original towers. For skyline views, Top of the Rock looks at the Empire State "
            "Building, which is the view most people actually want."
        ),
    },
    {
        "id": "newyork-food",
        "city": "New York",
        "country": "United States",
        "section": "food",
        "text": (
            "New York eating is defined by immigrant neighbourhoods: Flushing in Queens for "
            "regional Chinese, Jackson Heights for South Asian and Himalayan, Sunset Park "
            "for Mexican and Cantonese, Brighton Beach for Russian and Georgian. The "
            "canonical local foods remain the bagel, the slice, the pastrami sandwich and "
            "the halal cart platter. Reservations at the sought-after restaurants open "
            "exactly 30 days ahead at 10am."
        ),
    },
    {
        "id": "newyork-transport",
        "city": "New York",
        "country": "United States",
        "section": "transport",
        "text": (
            "The subway runs 24 hours, unusual among world metros, though late-night service "
            "reroutes for maintenance. OMNY contactless tap replaced the MetroCard and caps "
            "fares weekly. Express trains skip stops - check the letter or number and the "
            "colour of the dot on the platform sign. JFK connects via AirTrain to the E, J "
            "or Z lines; LaGuardia is bus-only and traffic-dependent."
        ),
    },
    {
        "id": "newyork-when",
        "city": "New York",
        "country": "United States",
        "section": "best time to visit",
        "text": (
            "September to early November is the best stretch: warm days, low humidity and "
            "the cultural season restarting. Late April to June is the spring equivalent. "
            "July and August are hot and humid with subway platforms far hotter than the "
            "street. December brings holiday crowds and the highest hotel rates of the year."
        ),
    },
]

# Cities the vector store can answer from, in canonical spelling.
SEEDED_CITIES: list[str] = ["Paris", "Tokyo", "New York"]

# Extra names the offline planner can recognise so the demo still routes
# sensibly without a live model. Not the same thing as being *in* the store.
KNOWN_CITY_NAMES: list[str] = [
    "New York",
    "New York City",
    "Paris",
    "Tokyo",
    "Kyoto",
    "Snohomish",
    "London",
    "Rome",
    "Sydney",
    "Cairo",
    "Reykjavik",
    "Barcelona",
    "Berlin",
    "Amsterdam",
    "Istanbul",
    "Bangkok",
    "Singapore",
    "Dubai",
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Lisbon",
    "Prague",
    "Vienna",
    "Seoul",
    "Osaka",
    "Marrakesh",
    "Cape Town",
    "Buenos Aires",
    "Mexico City",
    "Toronto",
    "Vancouver",
    "Chicago",
    "San Francisco",
    "Seattle",
    "Boston",
]


def normalise_city(name: str) -> str:
    """Canonical key for city comparison ('new york city' -> 'new york')."""
    cleaned = " ".join((name or "").strip().lower().replace(",", " ").split())
    aliases = {
        "new york city": "new york",
        "nyc": "new york",
        "ny": "new york",
        "tokyo japan": "tokyo",
        "paris france": "paris",
    }
    return aliases.get(cleaned, cleaned)
