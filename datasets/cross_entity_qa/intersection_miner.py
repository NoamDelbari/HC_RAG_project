"""
Multi-Property Intersection Miner for Cross-Entity Clusters

This module discovers entity clusters at the intersection of multiple properties,
enabling queries like "Nobel Prize winning physicists educated at Cambridge".

Phase A.2.3: Mine multi-property intersections for clusters
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations

from sparql_profiler import SPARQLProfiler, QueryResult, WIKIDATA_SPARQL_ENDPOINT
from constraint_strategies import (
    ConstraintStrategyBuilder,
    Constraint,
    CONTINENTS,
    MAJOR_COUNTRIES,
    ENTITY_TYPES,
    OCCUPATIONS,
    GENDERS
)


@dataclass
class IntersectionCluster:
    """Represents a cluster formed by property intersections."""
    name: str
    description: str
    properties: list
    constraints: list
    sparql: str
    entity_count: int = 0
    wikipedia_coverage: float = 0.0
    sample_entities: list = field(default_factory=list)
    viability_score: float = 0.0


class IntersectionMiner:
    """
    Mines multi-property intersections to discover viable entity clusters.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.profiler = SPARQLProfiler(endpoint)
        self.constraint_builder = ConstraintStrategyBuilder

    def build_intersection_query(
        self,
        base_type_qid: str,
        property_constraints: list,
        count_only: bool = True,
        limit: int = 100
    ) -> str:
        """
        Build a SPARQL query for entity intersection.

        Args:
            base_type_qid: Base entity type (e.g., Q5 for human)
            property_constraints: List of (property_id, value_qid) tuples or Constraint objects
            count_only: If True, return count; otherwise return entities
            limit: Maximum entities to return (when not count_only)
        """
        select_clause = "SELECT (COUNT(DISTINCT ?entity) AS ?count)" if count_only else """
SELECT DISTINCT ?entity ?entityLabel"""

        # Build constraint patterns
        constraint_patterns = []

        # Add base type constraint
        constraint_patterns.append(f"?entity wdt:P31/wdt:P279* wd:{base_type_qid} .")

        # Add property constraints
        for constraint in property_constraints:
            if isinstance(constraint, Constraint):
                constraint_patterns.append(constraint.sparql_pattern)
            elif isinstance(constraint, tuple) and len(constraint) == 2:
                prop_id, value_qid = constraint
                constraint_patterns.append(f"?entity wdt:{prop_id} wd:{value_qid} .")
            else:
                raise ValueError(f"Invalid constraint format: {constraint}")

        service_clause = "" if count_only else """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }"""

        limit_clause = "" if count_only else f"\nLIMIT {limit}"

        sparql = f"""{select_clause}
WHERE {{
  {chr(10).join('  ' + p for p in constraint_patterns)}{service_clause}
}}{limit_clause}"""

        return sparql

    def mine_award_occupation_intersection(
        self,
        award_qid: str,
        occupation_qid: str,
        country_qid: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None
    ) -> IntersectionCluster:
        """
        Find entities with a specific award AND occupation.

        Example: Nobel Prize winning physicists from Germany
        """
        constraints = []
        description_parts = []

        # Award constraint
        if year_start or year_end:
            award_pattern = f"""?entity p:P166 ?awardStmt .
  ?awardStmt ps:P166 wd:{award_qid} .
  ?awardStmt pq:P585 ?awardTime .
  BIND(YEAR(?awardTime) AS ?awardYear)"""
            if year_start and year_end:
                award_pattern += f"\n  FILTER(?awardYear >= {year_start} && ?awardYear <= {year_end})"
                description_parts.append(f"award {award_qid} ({year_start}-{year_end})")
            elif year_start:
                award_pattern += f"\n  FILTER(?awardYear >= {year_start})"
                description_parts.append(f"award {award_qid} (from {year_start})")
            else:
                award_pattern += f"\n  FILTER(?awardYear <= {year_end})"
                description_parts.append(f"award {award_qid} (until {year_end})")
            constraints.append(Constraint(
                constraint_type=None,
                property_id="P166",
                label="award_temporal",
                sparql_pattern=award_pattern,
                description=description_parts[-1]
            ))
        else:
            constraints.append(("P166", award_qid))
            description_parts.append(f"award {award_qid}")

        # Occupation constraint
        constraints.append(("P106", occupation_qid))
        description_parts.append(f"occupation {occupation_qid}")

        # Country constraint
        if country_qid:
            constraints.append(("P27", country_qid))
            description_parts.append(f"from {country_qid}")

        # Build and execute query
        sparql = self.build_intersection_query(
            base_type_qid=ENTITY_TYPES["human"],
            property_constraints=constraints,
            count_only=True
        )

        result = self.profiler.execute_query(sparql)

        name = f"award_{award_qid}_occ_{occupation_qid}"
        if country_qid:
            name += f"_country_{country_qid}"

        return IntersectionCluster(
            name=name,
            description=" + ".join(description_parts),
            properties=["P166", "P106"] + (["P27"] if country_qid else []),
            constraints=[str(c) for c in constraints],
            sparql=sparql,
            entity_count=result.count if result.success else 0
        )

    def mine_education_achievement_intersection(
        self,
        university_qid: str,
        achievement_property: str,
        achievement_value: str
    ) -> IntersectionCluster:
        """
        Find entities educated at a university with a specific achievement.

        Example: Nobel laureates educated at MIT
        """
        constraints = [
            ("P69", university_qid),  # educated at
            (achievement_property, achievement_value)
        ]

        sparql = self.build_intersection_query(
            base_type_qid=ENTITY_TYPES["human"],
            property_constraints=constraints,
            count_only=True
        )

        result = self.profiler.execute_query(sparql)

        return IntersectionCluster(
            name=f"educated_{university_qid}_{achievement_property}_{achievement_value}",
            description=f"Educated at {university_qid} with {achievement_property}={achievement_value}",
            properties=["P69", achievement_property],
            constraints=[str(c) for c in constraints],
            sparql=sparql,
            entity_count=result.count if result.success else 0
        )

    def mine_creative_role_intersection(
        self,
        work_qid: str,
        roles: list
    ) -> IntersectionCluster:
        """
        Find people with multiple roles in a creative work.

        Example: People who were both director AND writer of a film
        """
        role_properties = {
            "director": "P57",
            "writer": "P58",
            "producer": "P162",
            "composer": "P86",
            "cast": "P161"
        }

        # Build query that finds the work and its associated people
        role_patterns = []
        for role in roles:
            prop = role_properties.get(role)
            if prop:
                role_patterns.append(f"wd:{work_qid} wdt:{prop} ?entity .")

        if len(role_patterns) < 2:
            raise ValueError("Need at least 2 roles for intersection")

        # For intersection, we need entities that appear in ALL roles
        # Use FILTER to check entity appears in each role
        sparql = f"""SELECT (COUNT(DISTINCT ?entity) AS ?count)
WHERE {{
  {role_patterns[0]}
  {chr(10).join('  ' + p for p in role_patterns[1:])}
}}"""

        result = self.profiler.execute_query(sparql)

        return IntersectionCluster(
            name=f"work_{work_qid}_roles_{'_'.join(roles)}",
            description=f"Multiple roles ({', '.join(roles)}) in {work_qid}",
            properties=[role_properties[r] for r in roles if r in role_properties],
            constraints=[f"role={r}" for r in roles],
            sparql=sparql,
            entity_count=result.count if result.success else 0
        )

    def mine_geographic_categorical_intersection(
        self,
        entity_type_qid: str,
        country_qid: str,
        additional_property: Optional[str] = None,
        additional_value: Optional[str] = None
    ) -> IntersectionCluster:
        """
        Find entities of a type in a geographic region.

        Example: Universities in Germany, Companies in Japan with >10000 employees
        """
        constraints = [
            self.constraint_builder.categorical_instance_of(entity_type_qid),
            self.constraint_builder.geographic_country(country_qid)
        ]

        if additional_property and additional_value:
            constraints.append((additional_property, additional_value))

        sparql = self.build_intersection_query(
            base_type_qid=entity_type_qid,
            property_constraints=constraints[1:],  # Skip instance_of as it's the base
            count_only=True
        )

        result = self.profiler.execute_query(sparql)

        return IntersectionCluster(
            name=f"type_{entity_type_qid}_country_{country_qid}",
            description=f"Type {entity_type_qid} in {country_qid}",
            properties=["P31", "P17"] + ([additional_property] if additional_property else []),
            constraints=[str(c) for c in constraints],
            sparql=sparql,
            entity_count=result.count if result.success else 0
        )

    def mine_temporal_categorical_intersection(
        self,
        entity_type_qid: str,
        time_property: str,
        year_start: int,
        year_end: int,
        additional_constraint: Optional[Constraint] = None
    ) -> IntersectionCluster:
        """
        Find entities of a type within a time range.

        Example: Films released in the 1990s that won Academy Awards
        """
        temporal = self.constraint_builder.temporal_year_range(
            time_property=time_property,
            year_start=year_start,
            year_end=year_end
        )

        constraints = [temporal]
        if additional_constraint:
            constraints.append(additional_constraint)

        sparql = self.build_intersection_query(
            base_type_qid=entity_type_qid,
            property_constraints=constraints,
            count_only=True
        )

        result = self.profiler.execute_query(sparql)

        return IntersectionCluster(
            name=f"type_{entity_type_qid}_{year_start}_{year_end}",
            description=f"Type {entity_type_qid} from {year_start}-{year_end}",
            properties=["P31", time_property],
            constraints=[str(c) for c in constraints],
            sparql=sparql,
            entity_count=result.count if result.success else 0
        )


# =============================================================================
# PREDEFINED INTERSECTION QUERIES
# =============================================================================

INTERSECTION_TEMPLATES = {
    # Award + Occupation + Country
    "nobel_physics_german": {
        "type": "award_occupation",
        "params": {
            "award_qid": "Q38104",  # Nobel Prize in Physics
            "occupation_qid": "Q169470",  # physicist
            "country_qid": "Q183"  # Germany
        },
        "expected_k": (10, 30)
    },
    "nobel_chemistry_us": {
        "type": "award_occupation",
        "params": {
            "award_qid": "Q44585",  # Nobel Prize in Chemistry
            "occupation_qid": "Q593644",  # chemist
            "country_qid": "Q30"  # USA
        },
        "expected_k": (40, 70)
    },
    "oscar_best_director_british": {
        "type": "award_occupation",
        "params": {
            "award_qid": "Q103360",  # Academy Award for Best Director
            "occupation_qid": "Q2526255",  # film director
            "country_qid": "Q145"  # UK
        },
        "expected_k": (5, 15)
    },

    # Education + Achievement
    "cambridge_nobel": {
        "type": "education_achievement",
        "params": {
            "university_qid": "Q35794",  # Cambridge
            "achievement_property": "P166",
            "achievement_value": "Q7191"  # Nobel Prize
        },
        "expected_k": (80, 120)
    },
    "mit_turing": {
        "type": "education_achievement",
        "params": {
            "university_qid": "Q49108",  # MIT
            "achievement_property": "P166",
            "achievement_value": "Q80061"  # Turing Award
        },
        "expected_k": (15, 30)
    },
    "harvard_us_presidents": {
        "type": "education_achievement",
        "params": {
            "university_qid": "Q13371",  # Harvard
            "achievement_property": "P39",
            "achievement_value": "Q11696"  # US President
        },
        "expected_k": (5, 10)
    },

    # Geographic + Type
    "japanese_universities": {
        "type": "geographic_categorical",
        "params": {
            "entity_type_qid": "Q3918",  # university
            "country_qid": "Q17"  # Japan
        },
        "expected_k": (100, 200)
    },
    "german_car_companies": {
        "type": "geographic_categorical",
        "params": {
            "entity_type_qid": "Q786820",  # automobile manufacturer
            "country_qid": "Q183"  # Germany
        },
        "expected_k": (20, 50)
    },
    "french_museums": {
        "type": "geographic_categorical",
        "params": {
            "entity_type_qid": "Q33506",  # museum
            "country_qid": "Q142"  # France
        },
        "expected_k": (100, 300)
    },

    # Occupation + Gender + Country
    "female_us_astronauts": {
        "type": "award_occupation",  # Reusing template
        "params": {
            "award_qid": None,  # Special handling needed
            "occupation_qid": "Q11631",  # astronaut
            "country_qid": "Q30"  # USA
        },
        "custom_constraints": [
            ("P21", "Q6581072")  # female
        ],
        "expected_k": (40, 60)
    }
}


def run_intersection_mining(templates: list = None):
    """
    Run intersection mining for specified templates.

    Args:
        templates: List of template names (default: all)

    Returns:
        Dictionary of results
    """
    miner = IntersectionMiner()
    results = {}

    if templates is None:
        templates = list(INTERSECTION_TEMPLATES.keys())

    for template_name in templates:
        if template_name not in INTERSECTION_TEMPLATES:
            print(f"Unknown template: {template_name}")
            continue

        config = INTERSECTION_TEMPLATES[template_name]
        print(f"\nMining intersection: {template_name}")

        try:
            if config["type"] == "award_occupation":
                params = config["params"]
                if params.get("award_qid"):
                    cluster = miner.mine_award_occupation_intersection(
                        award_qid=params["award_qid"],
                        occupation_qid=params["occupation_qid"],
                        country_qid=params.get("country_qid")
                    )
                else:
                    # Custom query needed
                    continue

            elif config["type"] == "education_achievement":
                params = config["params"]
                cluster = miner.mine_education_achievement_intersection(
                    university_qid=params["university_qid"],
                    achievement_property=params["achievement_property"],
                    achievement_value=params["achievement_value"]
                )

            elif config["type"] == "geographic_categorical":
                params = config["params"]
                cluster = miner.mine_geographic_categorical_intersection(
                    entity_type_qid=params["entity_type_qid"],
                    country_qid=params["country_qid"]
                )

            else:
                print(f"  Unknown type: {config['type']}")
                continue

            results[template_name] = {
                "cluster": cluster,
                "expected_k": config.get("expected_k", (0, 1000))
            }

            expected_min, expected_max = config.get("expected_k", (0, 1000))
            status = "OK" if expected_min <= cluster.entity_count <= expected_max else "OUT_OF_RANGE"
            print(f"  -> {cluster.entity_count} entities ({status})")
            print(f"     Description: {cluster.description}")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results[template_name] = {"error": str(e)}

    return results


def discover_viable_intersections(
    min_k: int = 5,
    max_k: int = 50,
    sample_size: int = 20
):
    """
    Automatically discover viable intersection clusters within K bounds.

    Tries various combinations of properties to find bounded clusters.
    """
    miner = IntersectionMiner()
    viable_clusters = []

    # Define property value combinations to try
    awards = [
        ("Q7191", "Nobel Prize"),
        ("Q80061", "Turing Award"),
        ("Q19020", "Academy Award"),
        ("Q28835", "Fields Medal")
    ]

    occupations = [
        ("Q169470", "physicist"),
        ("Q593644", "chemist"),
        ("Q170790", "mathematician"),
        ("Q2526255", "film director"),
        ("Q36180", "writer")
    ]

    countries = [
        ("Q30", "USA"),
        ("Q145", "UK"),
        ("Q183", "Germany"),
        ("Q142", "France"),
        ("Q17", "Japan")
    ]

    print(f"Discovering viable intersections (K range: {min_k}-{max_k})...")
    print("=" * 60)

    tested = 0
    for award_qid, award_name in awards:
        for occ_qid, occ_name in occupations:
            for country_qid, country_name in countries:
                if tested >= sample_size:
                    break

                try:
                    cluster = miner.mine_award_occupation_intersection(
                        award_qid=award_qid,
                        occupation_qid=occ_qid,
                        country_qid=country_qid
                    )

                    tested += 1

                    if min_k <= cluster.entity_count <= max_k:
                        viable_clusters.append({
                            "name": f"{award_name} + {occ_name} from {country_name}",
                            "count": cluster.entity_count,
                            "cluster": cluster
                        })
                        print(f"VIABLE: {award_name} + {occ_name} from {country_name}: {cluster.entity_count}")

                except Exception as e:
                    tested += 1
                    continue

                time.sleep(0.5)  # Rate limiting

    print(f"\n{'=' * 60}")
    print(f"Tested: {tested} combinations")
    print(f"Viable clusters found: {len(viable_clusters)}")

    return viable_clusters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine multi-property intersections")
    parser.add_argument("--templates", nargs="+", help="Specific templates to mine")
    parser.add_argument("--discover", action="store_true", help="Auto-discover viable intersections")
    parser.add_argument("--min-k", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--max-k", type=int, default=50, help="Maximum cluster size")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    if args.discover:
        results = discover_viable_intersections(
            min_k=args.min_k,
            max_k=args.max_k
        )
    else:
        results = run_intersection_mining(templates=args.templates)

    if args.output:
        # Serialize results
        output_data = {}
        for name, data in results.items():
            if "error" in data:
                output_data[name] = {"error": data["error"]}
            elif "cluster" in data:
                cluster = data["cluster"]
                output_data[name] = {
                    "name": cluster.name,
                    "description": cluster.description,
                    "properties": cluster.properties,
                    "entity_count": cluster.entity_count,
                    "expected_k": data.get("expected_k", [0, 0])
                }
            elif "count" in data:
                output_data[name] = {
                    "name": data["name"],
                    "count": data["count"]
                }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")
