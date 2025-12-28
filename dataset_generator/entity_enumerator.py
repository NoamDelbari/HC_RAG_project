"""
Entity Enumerator for Cross-Entity Clusters

This module retrieves complete entity lists for viable clusters,
including QIDs, labels, and basic properties.

Phase A.3.1: Retrieve complete entity lists for viable clusters
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from sparql_profiler import SPARQLProfiler, WIKIDATA_SPARQL_ENDPOINT


@dataclass
class Entity:
    """Represents a Wikidata entity in a cluster."""
    qid: str
    label: str
    description: str = ""
    aliases: list = field(default_factory=list)
    wikipedia_title: Optional[str] = None
    properties: dict = field(default_factory=dict)


@dataclass
class ClusterEntities:
    """Complete entity list for a cluster."""
    cluster_id: str
    cluster_name: str
    cluster_type: str
    description: str
    entity_count: int
    entities: list = field(default_factory=list)
    binding_property: str = ""
    binding_value: str = ""
    constraints: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EntityEnumerator:
    """
    Retrieves complete entity lists for clusters from Wikidata.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.profiler = SPARQLProfiler(endpoint)

    def enumerate_award_winners(
        self,
        award_qid: str,
        award_name: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        limit: int = 500
    ) -> ClusterEntities:
        """
        Enumerate all winners of a specific award.

        Returns entities with QID, label, year of award, and Wikipedia link.
        """
        time_filter = ""
        if year_start and year_end:
            time_filter = f"""
    ?entity p:P166 ?statement .
    ?statement ps:P166 wd:{award_qid} .
    ?statement pq:P585 ?time .
    BIND(YEAR(?time) AS ?year)
    FILTER(?year >= {year_start} && ?year <= {year_end})"""
        elif year_start:
            time_filter = f"""
    ?entity p:P166 ?statement .
    ?statement ps:P166 wd:{award_qid} .
    ?statement pq:P585 ?time .
    BIND(YEAR(?time) AS ?year)
    FILTER(?year >= {year_start})"""
        elif year_end:
            time_filter = f"""
    ?entity p:P166 ?statement .
    ?statement ps:P166 wd:{award_qid} .
    ?statement pq:P585 ?time .
    BIND(YEAR(?time) AS ?year)
    FILTER(?year <= {year_end})"""
        else:
            time_filter = f"""
    ?entity p:P166 ?statement .
    ?statement ps:P166 wd:{award_qid} .
    OPTIONAL {{ ?statement pq:P585 ?time . BIND(YEAR(?time) AS ?year) }}"""

        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?year ?wikipedia WHERE {{
  {time_filter}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?year ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={"award_year": binding.get("year", "")}
                )
                entities.append(entity)

        description = f"Winners of {award_name}"
        if year_start and year_end:
            description += f" ({year_start}-{year_end})"

        return ClusterEntities(
            cluster_id=f"award_{award_qid}_{year_start or 'all'}_{year_end or 'all'}",
            cluster_name=f"{award_name} Winners",
            cluster_type="award_recognition",
            description=description,
            entity_count=len(entities),
            entities=entities,
            binding_property="P166",
            binding_value=award_qid,
            constraints=[f"year: {year_start}-{year_end}"] if year_start or year_end else []
        )

    def enumerate_cast_members(
        self,
        work_qid: str,
        work_name: str,
        limit: int = 200
    ) -> ClusterEntities:
        """
        Enumerate cast members of a film or TV show.
        """
        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?character ?characterLabel ?wikipedia WHERE {{
  wd:{work_qid} wdt:P161 ?entity .
  OPTIONAL {{
    wd:{work_qid} p:P161 ?castStatement .
    ?castStatement ps:P161 ?entity .
    ?castStatement pq:P453 ?character .
  }}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={
                        "character": binding.get("characterLabel", ""),
                        "character_qid": binding.get("character", "").split("/")[-1] if binding.get("character") else ""
                    }
                )
                entities.append(entity)

        return ClusterEntities(
            cluster_id=f"cast_{work_qid}",
            cluster_name=f"Cast of {work_name}",
            cluster_type="creative_ensemble",
            description=f"Cast members of {work_name}",
            entity_count=len(entities),
            entities=entities,
            binding_property="P161",
            binding_value=work_qid
        )

    def enumerate_filmography(
        self,
        director_qid: str,
        director_name: str,
        limit: int = 200
    ) -> ClusterEntities:
        """
        Enumerate films directed by a specific person.
        """
        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?year ?wikipedia WHERE {{
  ?entity wdt:P57 wd:{director_qid} .
  ?entity wdt:P31 wd:Q11424 .
  OPTIONAL {{ ?entity wdt:P577 ?date . BIND(YEAR(?date) AS ?year) }}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?year ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={"release_year": binding.get("year", "")}
                )
                entities.append(entity)

        return ClusterEntities(
            cluster_id=f"filmography_{director_qid}",
            cluster_name=f"Films by {director_name}",
            cluster_type="relational",
            description=f"Films directed by {director_name}",
            entity_count=len(entities),
            entities=entities,
            binding_property="P57",
            binding_value=director_qid
        )

    def enumerate_position_holders(
        self,
        position_qid: str,
        position_name: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        limit: int = 500
    ) -> ClusterEntities:
        """
        Enumerate holders of a specific position.
        """
        time_filter = ""
        if year_start or year_end:
            time_filter = """
  OPTIONAL { ?statement pq:P580 ?start . BIND(YEAR(?start) AS ?startYear) }
  OPTIONAL { ?statement pq:P582 ?end . BIND(YEAR(?end) AS ?endYear) }"""
            if year_start and year_end:
                time_filter += f"""
  FILTER((!BOUND(?endYear) || ?endYear >= {year_start}) && (!BOUND(?startYear) || ?startYear <= {year_end}))"""
            elif year_start:
                time_filter += f"""
  FILTER(!BOUND(?endYear) || ?endYear >= {year_start})"""
            elif year_end:
                time_filter += f"""
  FILTER(!BOUND(?startYear) || ?startYear <= {year_end})"""
        else:
            time_filter = """
  OPTIONAL { ?statement pq:P580 ?start . BIND(YEAR(?start) AS ?startYear) }
  OPTIONAL { ?statement pq:P582 ?end . BIND(YEAR(?end) AS ?endYear) }"""

        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?startYear ?endYear ?wikipedia WHERE {{
  ?entity p:P39 ?statement .
  ?statement ps:P39 wd:{position_qid} .
  {time_filter}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?startYear ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={
                        "start_year": binding.get("startYear", ""),
                        "end_year": binding.get("endYear", "")
                    }
                )
                entities.append(entity)

        description = f"Holders of {position_name}"
        if year_start and year_end:
            description += f" ({year_start}-{year_end})"

        return ClusterEntities(
            cluster_id=f"position_{position_qid}_{year_start or 'all'}_{year_end or 'all'}",
            cluster_name=position_name,
            cluster_type="organizational",
            description=description,
            entity_count=len(entities),
            entities=entities,
            binding_property="P39",
            binding_value=position_qid,
            constraints=[f"active: {year_start}-{year_end}"] if year_start or year_end else []
        )

    def enumerate_series_members(
        self,
        series_qid: str,
        series_name: str,
        limit: int = 200
    ) -> ClusterEntities:
        """
        Enumerate items in a series (film franchise, book series, etc).
        """
        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?ordinal ?date ?wikipedia WHERE {{
  ?entity wdt:P179 wd:{series_qid} .
  OPTIONAL {{ ?entity wdt:P1545 ?ordinal }}
  OPTIONAL {{ ?entity wdt:P577 ?date }}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?ordinal ?date ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={
                        "ordinal": binding.get("ordinal", ""),
                        "release_date": binding.get("date", "")
                    }
                )
                entities.append(entity)

        return ClusterEntities(
            cluster_id=f"series_{series_qid}",
            cluster_name=series_name,
            cluster_type="creative_ensemble",
            description=f"Items in the {series_name}",
            entity_count=len(entities),
            entities=entities,
            binding_property="P179",
            binding_value=series_qid
        )

    def enumerate_capitals(
        self,
        container_qid: str,
        container_name: str,
        region_type_qid: str = "Q6256",
        limit: int = 200
    ) -> ClusterEntities:
        """
        Enumerate capital cities of countries/regions in a container.
        """
        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?region ?regionLabel ?wikipedia WHERE {{
  ?region wdt:P36 ?entity .
  ?region wdt:P31 wd:{region_type_qid} .
  ?region wdt:P30 wd:{container_qid} .
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?regionLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={
                        "country": binding.get("regionLabel", ""),
                        "country_qid": binding.get("region", "").split("/")[-1]
                    }
                )
                entities.append(entity)

        return ClusterEntities(
            cluster_id=f"capitals_{container_qid}",
            cluster_name=f"Capital Cities of {container_name}",
            cluster_type="geographic_spatial",
            description=f"Capital cities of countries in {container_name}",
            entity_count=len(entities),
            entities=entities,
            binding_property="P36",
            binding_value=container_qid
        )

    def enumerate_event_participants(
        self,
        event_qid: str,
        event_name: str,
        participant_type_qid: Optional[str] = None,
        limit: int = 500
    ) -> ClusterEntities:
        """
        Enumerate participants in an event.
        """
        type_filter = ""
        if participant_type_qid:
            type_filter = f"?entity wdt:P31 wd:{participant_type_qid} ."

        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?wikipedia WHERE {{
  ?entity wdt:P1344 wd:{event_qid} .
  {type_filter}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title
                )
                entities.append(entity)

        return ClusterEntities(
            cluster_id=f"event_{event_qid}",
            cluster_name=f"{event_name} Participants",
            cluster_type="event_temporal",
            description=f"Participants in {event_name}",
            entity_count=len(entities),
            entities=entities,
            binding_property="P1344",
            binding_value=event_qid
        )

    def enumerate_founded_organizations(
        self,
        founder_qid: str,
        founder_name: str,
        limit: int = 100
    ) -> ClusterEntities:
        """
        Enumerate organizations founded by a person.
        """
        sparql = f"""
SELECT DISTINCT ?entity ?entityLabel ?entityDescription ?foundingDate ?wikipedia WHERE {{
  ?entity wdt:P112 wd:{founder_qid} .
  OPTIONAL {{ ?entity wdt:P571 ?foundingDate }}
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?foundingDate ?entityLabel
LIMIT {limit}
"""

        result = self.profiler.execute_query(sparql, timeout=120)

        entities = []
        if result.success:
            for binding in result.entities:
                qid = binding.get("entity", "").split("/")[-1]
                wiki_url = binding.get("wikipedia", "")
                wiki_title = wiki_url.split("/wiki/")[-1].replace("_", " ") if wiki_url else None

                entity = Entity(
                    qid=qid,
                    label=binding.get("entityLabel", ""),
                    description=binding.get("entityDescription", ""),
                    wikipedia_title=wiki_title,
                    properties={
                        "founding_date": binding.get("foundingDate", "")
                    }
                )
                entities.append(entity)

        return ClusterEntities(
            cluster_id=f"founded_by_{founder_qid}",
            cluster_name=f"Organizations Founded by {founder_name}",
            cluster_type="relational",
            description=f"Organizations founded by {founder_name}",
            entity_count=len(entities),
            entities=entities,
            binding_property="P112",
            binding_value=founder_qid
        )


def cluster_to_dict(cluster: ClusterEntities) -> dict:
    """Convert ClusterEntities to dictionary for JSON serialization."""
    return {
        "cluster_id": cluster.cluster_id,
        "cluster_name": cluster.cluster_name,
        "cluster_type": cluster.cluster_type,
        "description": cluster.description,
        "entity_count": cluster.entity_count,
        "binding_property": cluster.binding_property,
        "binding_value": cluster.binding_value,
        "constraints": cluster.constraints,
        "metadata": cluster.metadata,
        "entities": [
            {
                "qid": e.qid,
                "label": e.label,
                "description": e.description,
                "wikipedia_title": e.wikipedia_title,
                "properties": e.properties
            }
            for e in cluster.entities
        ]
    }


def save_cluster(cluster: ClusterEntities, output_dir: str):
    """Save cluster entities to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{cluster.cluster_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cluster_to_dict(cluster), f, indent=2, ensure_ascii=False)

    print(f"Saved {cluster.entity_count} entities to {filepath}")
    return filepath


# =============================================================================
# PREDEFINED CLUSTER ENUMERATION
# =============================================================================

CLUSTERS_TO_ENUMERATE = [
    # Award Recognition
    {"type": "award", "qid": "Q38104", "name": "Nobel Prize in Physics", "years": (2010, 2023)},
    {"type": "award", "qid": "Q80061", "name": "Turing Award", "years": None},
    {"type": "award", "qid": "Q103360", "name": "Academy Award for Best Director", "years": (2000, 2023)},

    # Creative Ensembles
    {"type": "cast", "qid": "Q47703", "name": "The Godfather"},
    {"type": "cast", "qid": "Q25188", "name": "Inception"},
    {"type": "cast", "qid": "Q44578", "name": "Titanic"},

    # Filmographies
    {"type": "filmography", "qid": "Q2001", "name": "Stanley Kubrick"},
    {"type": "filmography", "qid": "Q8877", "name": "Steven Spielberg"},
    {"type": "filmography", "qid": "Q55400", "name": "Hayao Miyazaki"},

    # Political Positions
    {"type": "position", "qid": "Q11696", "name": "President of the United States", "years": None},
    {"type": "position", "qid": "Q14211", "name": "Prime Minister of the United Kingdom", "years": None},
    {"type": "position", "qid": "Q144478", "name": "Secretary-General of the United Nations", "years": None},

    # Series
    {"type": "series", "qid": "Q642878", "name": "Marvel Cinematic Universe"},
    {"type": "series", "qid": "Q8337", "name": "Harry Potter"},
    {"type": "series", "qid": "Q2484680", "name": "James Bond films"},

    # Geographic
    {"type": "capitals", "qid": "Q46", "name": "Europe"},
    {"type": "capitals", "qid": "Q15", "name": "Africa"},

    # Founders
    {"type": "founded", "qid": "Q317521", "name": "Elon Musk"},
]


def enumerate_all_predefined(output_dir: str = "data/clusters"):
    """Enumerate all predefined clusters."""
    enumerator = EntityEnumerator()
    results = []

    for config in CLUSTERS_TO_ENUMERATE:
        print(f"\nEnumerating: {config['name']}...")

        try:
            if config["type"] == "award":
                years = config.get("years")
                cluster = enumerator.enumerate_award_winners(
                    award_qid=config["qid"],
                    award_name=config["name"],
                    year_start=years[0] if years else None,
                    year_end=years[1] if years else None
                )
            elif config["type"] == "cast":
                cluster = enumerator.enumerate_cast_members(
                    work_qid=config["qid"],
                    work_name=config["name"]
                )
            elif config["type"] == "filmography":
                cluster = enumerator.enumerate_filmography(
                    director_qid=config["qid"],
                    director_name=config["name"]
                )
            elif config["type"] == "position":
                years = config.get("years")
                cluster = enumerator.enumerate_position_holders(
                    position_qid=config["qid"],
                    position_name=config["name"],
                    year_start=years[0] if years else None,
                    year_end=years[1] if years else None
                )
            elif config["type"] == "series":
                cluster = enumerator.enumerate_series_members(
                    series_qid=config["qid"],
                    series_name=config["name"]
                )
            elif config["type"] == "capitals":
                cluster = enumerator.enumerate_capitals(
                    container_qid=config["qid"],
                    container_name=config["name"]
                )
            elif config["type"] == "founded":
                cluster = enumerator.enumerate_founded_organizations(
                    founder_qid=config["qid"],
                    founder_name=config["name"]
                )
            else:
                print(f"  Unknown type: {config['type']}")
                continue

            print(f"  Found {cluster.entity_count} entities")
            filepath = save_cluster(cluster, output_dir)
            results.append({
                "name": config["name"],
                "count": cluster.entity_count,
                "file": str(filepath)
            })

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "name": config["name"],
                "error": str(e)
            })

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enumerate entities in clusters")
    parser.add_argument("--output", type=str, default="data/clusters", help="Output directory")
    parser.add_argument("--cluster", type=str, help="Specific cluster to enumerate")

    args = parser.parse_args()

    if args.cluster:
        # Find and enumerate specific cluster
        for config in CLUSTERS_TO_ENUMERATE:
            if config["name"].lower() == args.cluster.lower():
                enumerator = EntityEnumerator()
                # ... enumerate specific cluster
                break
    else:
        # Enumerate all predefined clusters
        results = enumerate_all_predefined(args.output)

        print("\n" + "=" * 60)
        print("ENUMERATION SUMMARY")
        print("=" * 60)
        for r in results:
            if "error" in r:
                print(f"  {r['name']}: ERROR - {r['error']}")
            else:
                print(f"  {r['name']}: {r['count']} entities")
