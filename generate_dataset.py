"""
Synthetic Kickstarter dataset generator (v2).

IMPORTANT: this data is SYNTHETIC. It is NOT scraped from Kickstarter. Disclose this in
the README / write-up. Patterns are modelled on well-known trends in public Kickstarter
data (see the CS229 paper / Web Robots scrape), not copied from it:
  * larger goals fail more often
  * very long (or very short) campaigns do worse
  * category success rates differ widely (Dance/Theater/Comics high; Tech/Food/Journalism low)
  * polished, specific blurbs correlate with success; vague/hype blurbs with failure
  * a hidden "project quality" variable drives both blurb style and outcome, so text
    carries real (but noisy) signal and no model can reach ~100%

Schema is identical to the original CSV:
project_id, name_length, blurb_length, goal_usd, launch_date, deadline, duration_days,
currency, country, parent_category, category, location, location_type, blurb, outcome

Usage: python generate_dataset.py  ->  kickstarter_success_prediction_dataset.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N = 20000
TARGET_SUCCESS_RATE = 0.55
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- categories
# parent: (category effect on logit, mean ln(goal), {subcategory: [noun phrases]})
CATS = {
    "Art": (-0.10, 8.3, {
        "Mixed Media": ["a mixed media art series", "a large mixed media installation"],
        "Painting": ["a series of oil paintings", "a public painting exhibition"],
        "Illustration": ["an illustrated art collection", "a hand-drawn illustration series"],
        "Sculpture": ["a set of bronze sculptures", "an outdoor sculpture piece"]}),
    "Comics": (0.60, 8.0, {
        "Comic Books": ["a new comic book series", "the next issue of our comic"],
        "Graphic Novels": ["a full-length graphic novel", "a hardcover graphic novel"],
        "Webcomics": ["a webcomic collected into print", "a weekly webcomic anthology"]}),
    "Crafts": (-0.45, 7.6, {
        "Crochet": ["a crochet pattern collection", "handmade crochet kits"],
        "DIY": ["a DIY maker kit", "a do-it-yourself workshop set"],
        "Woodworking": ["hand-built wooden furniture", "a woodworking tool set"],
        "Candles": ["a line of soy candles", "small-batch scented candles"]}),
    "Dance": (0.90, 8.0, {
        "Dance Training": ["a community dance training program", "a youth dance academy"],
        "Performances": ["a live dance performance", "a touring dance show"],
        "Experimental": ["an experimental dance piece", "a contemporary dance residency"]}),
    "Design": (0.10, 9.5, {
        "Architecture": ["a modular architecture concept", "a community building design"],
        "Product Design": ["a new product design", "an everyday carry product"],
        "Graphic Design": ["a graphic design toolkit", "a typeface and poster set"]}),
    "Fashion": (-0.55, 9.0, {
        "Footwear": ["a sustainable footwear line", "a pair of everyday sneakers"],
        "Apparel": ["an ethical apparel collection", "a capsule clothing line"],
        "Accessories": ["a range of leather accessories", "a handmade jewelry collection"]}),
    "Film & Video": (0.05, 9.9, {
        "Short Film": ["an original short film", "a narrative short film"],
        "Documentary": ["a feature documentary", "a documentary about local communities"],
        "Webseries": ["a new webseries", "the second season of our webseries"],
        "Feature Film": ["an independent feature film", "a low-budget feature film"]}),
    "Food": (-0.65, 9.2, {
        "Drinks": ["a craft drink brand", "a small-batch cold brew line"],
        "Food": ["a new packaged food product", "a line of artisan sauces"],
        "Restaurants": ["a neighborhood restaurant", "a community food truck"]}),
    "Games": (0.10, 9.9, {
        "Tabletop Games": ["a tabletop board game", "a card game with custom art"],
        "Mobile Games": ["a mobile puzzle game", "a mobile strategy game"],
        "Video Games": ["an indie video game", "a story-driven video game"]}),
    "Journalism": (-0.85, 8.6, {
        "Print": ["an independent print magazine", "a local print newsletter"],
        "Audio": ["an investigative audio series", "a news podcast"],
        "Investigative": ["an investigative reporting project", "a long-form investigation"]}),
    "Music": (0.40, 8.7, {
        "Pop": ["a debut pop album", "a new pop EP"],
        "Rock": ["a rock album recorded live", "a new rock record"],
        "Hip-Hop": ["a hip-hop mixtape", "a debut hip-hop album"],
        "Classical": ["a classical recording", "a chamber music album"]}),
    "Photography": (-0.15, 8.3, {
        "Photobooks": ["a printed photobook", "a hardcover photobook"],
        "Nature": ["a nature photography project", "a wildlife photo series"],
        "Fine Art": ["a fine art photography series", "a gallery photo exhibition"]}),
    "Publishing": (-0.25, 8.6, {
        "Fiction": ["a debut fiction novel", "a fiction short story collection"],
        "Children's Books": ["an illustrated children's book", "a bilingual children's book"],
        "Nonfiction": ["a nonfiction book", "a narrative nonfiction title"]}),
    "Technology": (-0.95, 10.3, {
        "Apps": ["a productivity app", "a mobile app"],
        "Hardware": ["a new hardware device", "an open-source hardware board"],
        "Gadgets": ["a pocket-sized gadget", "a smart home gadget"],
        "Software": ["a desktop software tool", "an open-source software suite"]}),
    "Theater": (0.75, 8.4, {
        "Experimental Theater": ["an experimental theater production", "a devised theater piece"],
        "Musicals": ["an original musical", "a community musical"],
        "Plays": ["a new stage play", "a touring play"]}),
}

# ---------------------------------------------------------------- geography
LOCATIONS = {
    "US": ("USD", ["Austin", "Boston", "Chicago", "Los Angeles", "New York", "Portland", "San Francisco", "Seattle"]),
    "GB": ("GBP", ["Bristol", "Edinburgh", "London", "Manchester"]),
    "CA": ("CAD", ["Calgary", "Montreal", "Toronto", "Vancouver"]),
    "AU": ("AUD", ["Brisbane", "Melbourne", "Perth", "Sydney"]),
    "NZ": ("NZD", ["Auckland", "Christchurch", "Wellington"]),
    "DE": ("EUR", ["Berlin", "Cologne", "Hamburg", "Munich"]),
    "FR": ("EUR", ["Lyon", "Marseille", "Paris", "Toulouse"]),
    "ES": ("EUR", ["Barcelona", "Madrid", "Seville", "Valencia"]),
    "IT": ("EUR", ["Florence", "Milan", "Rome", "Turin"]),
    "NL": ("EUR", ["Amsterdam", "Rotterdam", "Utrecht"]),
    "IE": ("EUR", ["Cork", "Dublin", "Galway"]),
    "SE": ("SEK", ["Gothenburg", "Malmo", "Stockholm"]),
    "DK": ("DKK", ["Aarhus", "Copenhagen", "Odense"]),
    "NO": ("NOK", ["Bergen", "Oslo", "Trondheim"]),
    "IN": ("INR", ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai"]),
}
COUNTRY_P = {"US": .46, "GB": .11, "CA": .07, "AU": .06, "NZ": .02, "DE": .05, "FR": .04, "ES": .03,
             "IT": .03, "NL": .03, "IE": .015, "SE": .02, "DK": .02, "NO": .015, "IN": .04}
COUNTRY_EFFECT = {"US": .20, "GB": .10, "CA": .05, "IN": -.35}
LOC_TYPES = ["City", "Suburb", "Town", "LocalAdmin", "County", "Zip", "Island"]
LOC_P = [.55, .11, .10, .09, .08, .05, .02]
LOC_EFFECT = {"City": .10, "Zip": -.10, "Island": -.05}

# ---------------------------------------------------------------- text pools
STRONG = ["with a working prototype already tested", "backed by an experienced team", "with a clear production plan",
          "after two years of careful development", "with early supporters already onboard",
          "with a detailed budget and timeline", "featuring a professional collaborator",
          "with finished samples ready to ship", "and a fully designed first edition",
          "supported by a local partner"]
WEAK = ["we hope to make it happen someday", "this could be amazing if funded", "help us because we need money",
        "the best and most revolutionary idea ever", "it is just an early idea for now",
        "please support our first attempt", "a game changing concept with no plan yet",
        "we are still figuring out the details", "trust us, it will be incredible",
        "something totally unique and amazing"]
NEUTRAL = ["for a wider audience", "for our community", "for people who love the craft",
           "to share with the world", "built around an original idea", "inspired by everyday life"]
OPENERS = ["We are creating", "Help us launch", "Support", "Join us in making", "Introducing", "Bringing to life"]


def make_blurb(phrase, q):
    p_strong = 1 / (1 + np.exp(-1.6 * q))
    parts = [f"{rng.choice(OPENERS)} {phrase}"]
    for _ in range(rng.integers(1, 3)):                       # 1-2 markers
        pool = STRONG if rng.random() < p_strong else WEAK
        parts.append(rng.choice(pool))
    if rng.random() < 0.5:
        parts.append(rng.choice(NEUTRAL))
    text = parts[0] + ", " + ", ".join(parts[1:])
    return text + "."


# ---------------------------------------------------------------- generate
parents = list(CATS)
parent_idx = rng.choice(len(parents), N)
rows = []
sub_jitter = {(p, s): rng.normal(0, .25) for p in CATS for s in CATS[p][2]}

for i in range(N):
    parent = parents[parent_idx[i]]
    cat_eff, mu_goal, subs = CATS[parent]
    sub = rng.choice(list(subs))
    phrase = rng.choice(subs[sub])

    country = rng.choice(list(COUNTRY_P), p=np.array(list(COUNTRY_P.values())) / sum(COUNTRY_P.values()))
    currency, cities = LOCATIONS[country]
    location = rng.choice(cities)
    loc_type = rng.choice(LOC_TYPES, p=LOC_P)

    goal = float(np.clip(np.exp(rng.normal(mu_goal, 1.25)), 50, 2_000_000))
    duration = int(np.clip(round(rng.normal(34, 11)), 15, 60))
    launch = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 1186)))
    deadline = launch + pd.Timedelta(days=duration)
    name_len = int(np.clip(round(rng.normal(34, 13)), 5, 60))

    q = rng.normal()                       # hidden project quality (not in the CSV)
    blurb = make_blurb(phrase, q)
    blen = len(blurb)

    logit = (cat_eff + sub_jitter[(parent, sub)]
             - 0.62 * (np.log(goal) - mu_goal)                       # bigger goal (vs. category norm) -> worse
             - 0.35 * (np.log(goal) - 9.0)                           # bigger goal overall -> worse
             - 0.035 * max(duration - 35, 0) - 0.03 * max(20 - duration, 0)
             - 0.012 * abs(name_len - 30)
             - 0.004 * abs(blen - 80)
             + (0.20 if launch.month in (1, 3, 9, 10) else 0) - (0.30 if launch.month == 12 else 0)
             - (0.15 if launch.dayofweek >= 5 else 0)
             + COUNTRY_EFFECT.get(country, 0) + LOC_EFFECT.get(loc_type, 0)
             + 0.85 * q)
    rows.append(dict(name_length=name_len, blurb_length=blen, goal_usd=round(goal, 2),
                     launch_date=launch.date().isoformat(), deadline=deadline.date().isoformat(),
                     duration_days=duration, currency=currency, country=country, parent_category=parent,
                     category=sub, location=location, location_type=loc_type, blurb=blurb, _logit=logit))

df = pd.DataFrame(rows)

# calibrate intercept so overall success rate hits the target, then sample the outcome
lo, hi = -6.0, 6.0
for _ in range(60):
    b = (lo + hi) / 2
    if (1 / (1 + np.exp(-(df["_logit"] + b)))).mean() > TARGET_SUCCESS_RATE:
        hi = b
    else:
        lo = b
df["outcome"] = (rng.random(N) < 1 / (1 + np.exp(-(df["_logit"] + b)))).astype(int)
df = df.drop(columns="_logit")
df.insert(0, "project_id", np.arange(1, N + 1))

df.to_csv("kickstarter_success_prediction_dataset.csv", index=False)
print(df.shape, "success rate:", round(df.outcome.mean(), 3), "unique blurbs:", df.blurb.nunique())
