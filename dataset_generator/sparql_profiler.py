"""
SPARQL Profiler for Cross-Entity Cluster Discovery

This module provides SPARQL query building and execution for discovering
and profiling entity clusters based on Wikidata properties.

Phase A.2.1: Single-property cluster enumeration queries
"""

import json
import time
import requests
from typing import Optional
from dataclasses import dataclass, field


WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_TIMEOUT = 60
REQUEST_DELAY = 1.0  # Seconds between requests to respect rate limits


@dataclass
class ClusterQuery:
    """Represents a SPARQL query for cluster discovery."""
    name: str
    description: str
    sparql: str
    cluster_type: str
    primary_property: str
    constraints: list = field(default_factory=list)
    expected_k_range: tuple = (5, 50)


@dataclass
class QueryResult:
    """Results from a SPARQL query execution."""
    success: bool
    count: int = 0
    entities: list = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0


class SPARQLProfiler:
    """
    Builds and executes SPARQL queries for cross-entity cluster profiling.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.endpoint = endpoint
        self.last_request_time = 0

    def _throttle(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def execute_query(self, sparql: str, timeout: int = DEFAULT_TIMEOUT) -> QueryResult:
        """
        Execute a SPARQL query against Wikidata.

        Args:
            sparql: The SPARQL query string
            timeout: Request timeout in seconds

        Returns:
            QueryResult with success status, count, and entities
        """
        self._throttle()

        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "CrossEntityClusterProfiler/1.0 (Research Project)"
        }

        start_time = time.time()

        try:
            response = requests.get(
                self.endpoint,
                params={"query": sparql, "format": "json"},
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])

            # Check if this is a COUNT query
            if bindings and "count" in bindings[0]:
                count = int(bindings[0]["count"]["value"])
                return QueryResult(
                    success=True,
                    count=count,
                    execution_time=time.time() - start_time
                )

            # Otherwise, extract entities
            entities = []
            for binding in bindings:
                entity = {}
                for key, value in binding.items():
                    entity[key] = value.get("value", "")
                entities.append(entity)

            return QueryResult(
                success=True,
                count=len(entities),
                entities=entities,
                execution_time=time.time() - start_time
            )

        except requests.exceptions.Timeout:
            return QueryResult(
                success=False,
                error="Query timeout",
                execution_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return QueryResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
        except (json.JSONDecodeError, KeyError) as e:
            return QueryResult(
                success=False,
                error=f"Parse error: {e}",
                execution_time=time.time() - start_time
            )

    # =========================================================================
    # AWARD RECOGNITION QUERIES (P166)
    # =========================================================================

    def build_award_winners_query(
        self,
        award_qid: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for entities who received a specific award.

        Args:
            award_qid: Wikidata QID of the award (e.g., Q7191 for Nobel Prize)
            year_start: Optional start year filter
            year_end: Optional end year filter
            count_only: If True, return COUNT query; otherwise return full entities
        """
        select_clause = "SELECT (COUNT(DISTINCT ?entity) AS ?count)" if count_only else """
SELECT DISTINCT ?entity ?entityLabel ?year WHERE"""

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
  ?entity wdt:P166 wd:{award_qid} ."""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{{time_filter}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?year"

        constraints = []
        if year_start:
            constraints.append(f"year >= {year_start}")
        if year_end:
            constraints.append(f"year <= {year_end}")

        return ClusterQuery(
            name=f"award_winners_{award_qid}",
            description=f"Winners of award {award_qid}" + (f" ({year_start}-{year_end})" if year_start or year_end else ""),
            sparql=sparql,
            cluster_type="award_recognition",
            primary_property="P166",
            constraints=constraints
        )

    # =========================================================================
    # CREATIVE ENSEMBLE QUERIES (P161 - cast member)
    # =========================================================================

    def build_cast_query(
        self,
        work_qid: str,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for cast members of a film/TV show.

        Args:
            work_qid: Wikidata QID of the creative work (film, TV series)
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?actor) AS ?count)" if count_only else """
SELECT DISTINCT ?actor ?actorLabel ?character ?characterLabel WHERE"""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        character_clause = "" if count_only else """
  OPTIONAL {
    ?work p:P161 ?castStatement .
    ?castStatement ps:P161 ?actor .
    ?castStatement pq:P453 ?character .
  }"""

        sparql = f"""{select_clause}
{{
  wd:{work_qid} wdt:P161 ?actor .{character_clause}{service_clause}
}}"""

        return ClusterQuery(
            name=f"cast_{work_qid}",
            description=f"Cast members of {work_qid}",
            sparql=sparql,
            cluster_type="creative_ensemble",
            primary_property="P161"
        )

    # =========================================================================
    # RELATIONAL QUERIES (P57 - director filmography)
    # =========================================================================

    def build_director_filmography_query(
        self,
        director_qid: str,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for films directed by a specific person.

        Args:
            director_qid: Wikidata QID of the director
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?film) AS ?count)" if count_only else """
SELECT DISTINCT ?film ?filmLabel ?year WHERE"""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        year_clause = "" if count_only else """
  OPTIONAL { ?film wdt:P577 ?date . BIND(YEAR(?date) AS ?year) }"""

        sparql = f"""{select_clause}
{{
  ?film wdt:P57 wd:{director_qid} .
  ?film wdt:P31 wd:Q11424 .{year_clause}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?year"

        return ClusterQuery(
            name=f"filmography_{director_qid}",
            description=f"Films directed by {director_qid}",
            sparql=sparql,
            cluster_type="relational",
            primary_property="P57"
        )

    # =========================================================================
    # ORGANIZATIONAL QUERIES (P39 - position held)
    # =========================================================================

    def build_position_holders_query(
        self,
        position_qid: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for people who held a specific position.

        Args:
            position_qid: Wikidata QID of the position (e.g., Q11696 for US President)
            year_start: Optional start year filter
            year_end: Optional end year filter
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?person) AS ?count)" if count_only else """
SELECT DISTINCT ?person ?personLabel ?startYear ?endYear WHERE"""

        time_filter = ""
        if year_start or year_end:
            time_filter = """
  OPTIONAL { ?statement pq:P580 ?start . BIND(YEAR(?start) AS ?startYear) }
  OPTIONAL { ?statement pq:P582 ?end . BIND(YEAR(?end) AS ?endYear) }"""
            if year_start and year_end:
                time_filter += f"""
  FILTER(!BOUND(?endYear) || ?endYear >= {year_start})
  FILTER(!BOUND(?startYear) || ?startYear <= {year_end})"""
            elif year_start:
                time_filter += f"""
  FILTER(!BOUND(?endYear) || ?endYear >= {year_start})"""
            elif year_end:
                time_filter += f"""
  FILTER(!BOUND(?startYear) || ?startYear <= {year_end})"""
        else:
            time_filter = """
  OPTIONAL { ?statement pq:P580 ?start . BIND(YEAR(?start) AS ?startYear) }
  OPTIONAL { ?statement pq:P582 ?end . BIND(YEAR(?end) AS ?endYear) }""" if not count_only else ""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{
  ?person p:P39 ?statement .
  ?statement ps:P39 wd:{position_qid} .{time_filter}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?startYear"

        constraints = []
        if year_start:
            constraints.append(f"active >= {year_start}")
        if year_end:
            constraints.append(f"active <= {year_end}")

        return ClusterQuery(
            name=f"position_{position_qid}",
            description=f"Holders of position {position_qid}",
            sparql=sparql,
            cluster_type="organizational",
            primary_property="P39",
            constraints=constraints
        )

    # =========================================================================
    # GEOGRAPHIC QUERIES (P36 - capital)
    # =========================================================================

    def build_capitals_query(
        self,
        region_type_qid: Optional[str] = None,
        container_qid: Optional[str] = None,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for capital cities.

        Args:
            region_type_qid: Type of region (e.g., Q6256 for country)
            container_qid: Container region (e.g., Q46 for Europe)
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?capital) AS ?count)" if count_only else """
SELECT DISTINCT ?capital ?capitalLabel ?region ?regionLabel WHERE"""

        type_filter = ""
        if region_type_qid:
            type_filter = f"""
  ?region wdt:P31 wd:{region_type_qid} ."""

        container_filter = ""
        if container_qid:
            container_filter = f"""
  ?region wdt:P30 wd:{container_qid} ."""  # P30 = continent

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{
  ?region wdt:P36 ?capital .{type_filter}{container_filter}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?regionLabel"

        constraints = []
        if region_type_qid:
            constraints.append(f"region_type = {region_type_qid}")
        if container_qid:
            constraints.append(f"contained_in = {container_qid}")

        return ClusterQuery(
            name="capitals",
            description="Capital cities" + (f" of {container_qid}" if container_qid else ""),
            sparql=sparql,
            cluster_type="geographic_spatial",
            primary_property="P36",
            constraints=constraints
        )

    # =========================================================================
    # SERIES QUERIES (P179 - part of series)
    # =========================================================================

    def build_series_members_query(
        self,
        series_qid: str,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for items in a series (film franchise, book series, etc).

        Args:
            series_qid: Wikidata QID of the series
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?item) AS ?count)" if count_only else """
SELECT DISTINCT ?item ?itemLabel ?ordinal ?date WHERE"""

        ordinal_clause = "" if count_only else """
  OPTIONAL { ?item wdt:P1545 ?ordinal }
  OPTIONAL { ?item wdt:P577 ?date }"""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{
  ?item wdt:P179 wd:{series_qid} .{ordinal_clause}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?ordinal ?date"

        return ClusterQuery(
            name=f"series_{series_qid}",
            description=f"Items in series {series_qid}",
            sparql=sparql,
            cluster_type="creative_ensemble",
            primary_property="P179"
        )

    # =========================================================================
    # EVENT PARTICIPATION QUERIES (P1344)
    # =========================================================================

    def build_event_participants_query(
        self,
        event_qid: str,
        participant_type_qid: Optional[str] = None,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for participants in an event.

        Args:
            event_qid: Wikidata QID of the event
            participant_type_qid: Optional type filter (e.g., Q5 for humans)
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?participant) AS ?count)" if count_only else """
SELECT DISTINCT ?participant ?participantLabel WHERE"""

        type_filter = ""
        if participant_type_qid:
            type_filter = f"""
  ?participant wdt:P31 wd:{participant_type_qid} ."""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{
  ?participant wdt:P1344 wd:{event_qid} .{type_filter}{service_clause}
}}"""

        return ClusterQuery(
            name=f"event_{event_qid}",
            description=f"Participants in event {event_qid}",
            sparql=sparql,
            cluster_type="event_temporal",
            primary_property="P1344"
        )

    # =========================================================================
    # FOUNDER QUERIES (P112)
    # =========================================================================

    def build_founded_by_query(
        self,
        founder_qid: str,
        org_type_qid: Optional[str] = None,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for organizations founded by a person.

        Args:
            founder_qid: Wikidata QID of the founder
            org_type_qid: Optional organization type filter
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?org) AS ?count)" if count_only else """
SELECT DISTINCT ?org ?orgLabel ?foundingDate WHERE"""

        type_filter = ""
        if org_type_qid:
            type_filter = f"""
  ?org wdt:P31/wdt:P279* wd:{org_type_qid} ."""

        date_clause = "" if count_only else """
  OPTIONAL { ?org wdt:P571 ?foundingDate }"""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{
  ?org wdt:P112 wd:{founder_qid} .{type_filter}{date_clause}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?foundingDate"

        return ClusterQuery(
            name=f"founded_by_{founder_qid}",
            description=f"Organizations founded by {founder_qid}",
            sparql=sparql,
            cluster_type="relational",
            primary_property="P112"
        )

    # =========================================================================
    # AUTHOR WORKS QUERIES (P50)
    # =========================================================================

    def build_author_works_query(
        self,
        author_qid: str,
        work_type_qid: Optional[str] = None,
        count_only: bool = False
    ) -> ClusterQuery:
        """
        Build query for works by an author.

        Args:
            author_qid: Wikidata QID of the author
            work_type_qid: Optional work type filter (e.g., Q7725634 for literary work)
            count_only: If True, return COUNT query
        """
        select_clause = "SELECT (COUNT(DISTINCT ?work) AS ?count)" if count_only else """
SELECT DISTINCT ?work ?workLabel ?pubDate WHERE"""

        type_filter = ""
        if work_type_qid:
            type_filter = f"""
  ?work wdt:P31/wdt:P279* wd:{work_type_qid} ."""

        date_clause = "" if count_only else """
  OPTIONAL { ?work wdt:P577 ?pubDate }"""

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        sparql = f"""{select_clause}
{{
  ?work wdt:P50 wd:{author_qid} .{type_filter}{date_clause}{service_clause}
}}"""

        if not count_only:
            sparql += "\nORDER BY ?pubDate"

        return ClusterQuery(
            name=f"works_by_{author_qid}",
            description=f"Works by author {author_qid}",
            sparql=sparql,
            cluster_type="relational",
            primary_property="P50"
        )

    # =========================================================================
    # WIKIPEDIA COVERAGE CHECK
    # =========================================================================

    def build_wikipedia_coverage_query(
        self,
        base_query: ClusterQuery
    ) -> ClusterQuery:
        """
        Modify a query to check Wikipedia coverage.

        Returns count of entities that have English Wikipedia articles.
        """
        # Extract the WHERE clause content from the base query
        sparql = base_query.sparql

        # Build coverage query
        coverage_sparql = f"""SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE {{
  {{
    {sparql.split('WHERE')[1].split('}}')[0] if 'WHERE' in sparql else sparql}
  }}
  ?article schema:about ?entity .
  ?article schema:isPartOf <https://en.wikipedia.org/> .
}}"""

        return ClusterQuery(
            name=f"{base_query.name}_wiki_coverage",
            description=f"Wikipedia coverage for {base_query.description}",
            sparql=coverage_sparql,
            cluster_type=base_query.cluster_type,
            primary_property=base_query.primary_property,
            constraints=base_query.constraints + ["has_en_wikipedia"]
        )


# =============================================================================
# PREDEFINED CLUSTER QUERIES
# =============================================================================

PREDEFINED_CLUSTERS = {
    # Award Recognition
    "nobel_physics_2010s": {
        "method": "build_award_winners_query",
        "args": {"award_qid": "Q38104", "year_start": 2010, "year_end": 2019},
        "expected_k": (10, 15)
    },
    "nobel_literature_all": {
        "method": "build_award_winners_query",
        "args": {"award_qid": "Q37922"},
        "expected_k": (100, 120)
    },
    "oscar_best_picture_2010s": {
        "method": "build_award_winners_query",
        "args": {"award_qid": "Q102427", "year_start": 2010, "year_end": 2019},
        "expected_k": (10, 10)
    },
    "turing_award_all": {
        "method": "build_award_winners_query",
        "args": {"award_qid": "Q80061"},
        "expected_k": (70, 80)
    },

    # Creative Ensembles
    "cast_godfather": {
        "method": "build_cast_query",
        "args": {"work_qid": "Q47703"},
        "expected_k": (15, 25)
    },
    "cast_inception": {
        "method": "build_cast_query",
        "args": {"work_qid": "Q25188"},
        "expected_k": (10, 20)
    },

    # Director Filmographies
    "kubrick_films": {
        "method": "build_director_filmography_query",
        "args": {"director_qid": "Q2001"},
        "expected_k": (12, 16)
    },
    "spielberg_films": {
        "method": "build_director_filmography_query",
        "args": {"director_qid": "Q8877"},
        "expected_k": (30, 40)
    },
    "miyazaki_films": {
        "method": "build_director_filmography_query",
        "args": {"director_qid": "Q55400"},
        "expected_k": (10, 15)
    },

    # Political Positions
    "us_presidents": {
        "method": "build_position_holders_query",
        "args": {"position_qid": "Q11696"},
        "expected_k": (45, 50)
    },
    "uk_prime_ministers": {
        "method": "build_position_holders_query",
        "args": {"position_qid": "Q14211"},
        "expected_k": (55, 60)
    },
    "un_secretaries_general": {
        "method": "build_position_holders_query",
        "args": {"position_qid": "Q144478"},
        "expected_k": (9, 10)
    },

    # Geographic
    "european_capitals": {
        "method": "build_capitals_query",
        "args": {"region_type_qid": "Q6256", "container_qid": "Q46"},
        "expected_k": (40, 50)
    },
    "african_capitals": {
        "method": "build_capitals_query",
        "args": {"region_type_qid": "Q6256", "container_qid": "Q15"},
        "expected_k": (50, 60)
    },

    # Series
    "mcu_films": {
        "method": "build_series_members_query",
        "args": {"series_qid": "Q642878"},
        "expected_k": (30, 40)
    },
    "harry_potter_books": {
        "method": "build_series_members_query",
        "args": {"series_qid": "Q8337"},
        "expected_k": (7, 10)
    },
    "james_bond_films": {
        "method": "build_series_members_query",
        "args": {"series_qid": "Q2484680"},
        "expected_k": (25, 30)
    },

    # Founders
    "musk_companies": {
        "method": "build_founded_by_query",
        "args": {"founder_qid": "Q317521"},
        "expected_k": (5, 10)
    },

    # Authors
    "shakespeare_plays": {
        "method": "build_author_works_query",
        "args": {"author_qid": "Q692", "work_type_qid": "Q25379"},
        "expected_k": (35, 45)
    },
    "asimov_novels": {
        "method": "build_author_works_query",
        "args": {"author_qid": "Q34981", "work_type_qid": "Q7725634"},
        "expected_k": (40, 60)
    }
}


def run_cluster_profiling(clusters: list = None, count_only: bool = True):
    """
    Run profiling queries for specified clusters.

    Args:
        clusters: List of cluster names to profile (default: all predefined)
        count_only: If True, only get counts; otherwise get full entity lists

    Returns:
        Dictionary of cluster names to QueryResult objects
    """
    profiler = SPARQLProfiler()
    results = {}

    if clusters is None:
        clusters = list(PREDEFINED_CLUSTERS.keys())

    for cluster_name in clusters:
        if cluster_name not in PREDEFINED_CLUSTERS:
            print(f"Unknown cluster: {cluster_name}")
            continue

        config = PREDEFINED_CLUSTERS[cluster_name]
        method = getattr(profiler, config["method"])
        args = config["args"].copy()
        args["count_only"] = count_only

        query = method(**args)
        print(f"Profiling {cluster_name}...")

        result = profiler.execute_query(query.sparql)
        results[cluster_name] = {
            "query": query,
            "result": result,
            "expected_k": config["expected_k"]
        }

        if result.success:
            expected_min, expected_max = config["expected_k"]
            status = "OK" if expected_min <= result.count <= expected_max else "OUT_OF_RANGE"
            print(f"  -> {result.count} entities ({status}, expected {expected_min}-{expected_max})")
        else:
            print(f"  -> ERROR: {result.error}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile cross-entity clusters via SPARQL")
    parser.add_argument("--clusters", nargs="+", help="Specific clusters to profile")
    parser.add_argument("--full", action="store_true", help="Get full entity lists (not just counts)")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    results = run_cluster_profiling(
        clusters=args.clusters,
        count_only=not args.full
    )

    if args.output:
        # Serialize results
        output_data = {}
        for name, data in results.items():
            output_data[name] = {
                "description": data["query"].description,
                "cluster_type": data["query"].cluster_type,
                "primary_property": data["query"].primary_property,
                "count": data["result"].count,
                "success": data["result"].success,
                "error": data["result"].error,
                "expected_k": data["expected_k"],
                "entities": data["result"].entities if data["result"].entities else []
            }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "="*60)
    print("PROFILING SUMMARY")
    print("="*60)

    successful = sum(1 for d in results.values() if d["result"].success)
    in_range = sum(1 for d in results.values()
                   if d["result"].success and
                   d["expected_k"][0] <= d["result"].count <= d["expected_k"][1])

    print(f"Total clusters profiled: {len(results)}")
    print(f"Successful queries: {successful}/{len(results)}")
    print(f"Within expected K range: {in_range}/{successful}")
