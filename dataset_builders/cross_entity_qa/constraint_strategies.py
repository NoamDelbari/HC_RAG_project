"""
Constraint Strategies for Cross-Entity Cluster Formation

This module provides constraint strategies to bound cluster sizes to target K ranges.
Implements temporal, geographic, threshold, and categorical constraints.

Phase A.2.2: Constraint strategy implementation
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ConstraintType(Enum):
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    THRESHOLD = "threshold"
    CATEGORICAL = "categorical"
    COMPOSITE = "composite"


@dataclass
class Constraint:
    """Represents a constraint that can be applied to bound cluster size."""
    constraint_type: ConstraintType
    property_id: str
    label: str
    sparql_pattern: str
    description: str
    parameters: dict = field(default_factory=dict)


@dataclass
class ConstraintApplication:
    """Records how a constraint was applied to achieve target K."""
    constraint: Constraint
    parameter_values: dict
    resulting_k: int
    sparql_fragment: str


class ConstraintStrategyBuilder:
    """
    Builds SPARQL constraint fragments to bound cluster sizes.
    """

    # =========================================================================
    # TEMPORAL CONSTRAINTS
    # =========================================================================

    @staticmethod
    def temporal_year_range(
        time_property: str = "P585",
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        use_qualifier: bool = False,
        statement_var: str = "statement"
    ) -> Constraint:
        """
        Create a temporal constraint filtering by year range.

        Args:
            time_property: Wikidata property for time (P585=point in time, P580=start, P582=end)
            year_start: Minimum year (inclusive)
            year_end: Maximum year (inclusive)
            use_qualifier: If True, use qualifier syntax (pq:); otherwise direct (wdt:)
            statement_var: Variable name for statement (used with qualifiers)
        """
        if use_qualifier:
            # Qualifier pattern: ?statement pq:P585 ?time
            base_pattern = f"?{statement_var} pq:{time_property} ?time ."
        else:
            # Direct pattern: ?entity wdt:P585 ?time
            base_pattern = f"?entity wdt:{time_property} ?time ."

        filters = []
        if year_start:
            filters.append(f"YEAR(?time) >= {year_start}")
        if year_end:
            filters.append(f"YEAR(?time) <= {year_end}")

        filter_clause = f"FILTER({' && '.join(filters)})" if filters else ""

        sparql_pattern = f"""
  {base_pattern}
  {filter_clause}"""

        description = "Temporal filter"
        if year_start and year_end:
            description = f"Years {year_start}-{year_end}"
        elif year_start:
            description = f"From {year_start} onwards"
        elif year_end:
            description = f"Up to {year_end}"

        return Constraint(
            constraint_type=ConstraintType.TEMPORAL,
            property_id=time_property,
            label="year_range",
            sparql_pattern=sparql_pattern.strip(),
            description=description,
            parameters={"year_start": year_start, "year_end": year_end}
        )

    @staticmethod
    def temporal_decade(decade: int, time_property: str = "P585", use_qualifier: bool = False) -> Constraint:
        """Create constraint for a specific decade (e.g., 1990 for 1990s)."""
        return ConstraintStrategyBuilder.temporal_year_range(
            time_property=time_property,
            year_start=decade,
            year_end=decade + 9,
            use_qualifier=use_qualifier
        )

    @staticmethod
    def temporal_century(century: int, time_property: str = "P585") -> Constraint:
        """Create constraint for a specific century (e.g., 20 for 20th century)."""
        year_start = (century - 1) * 100 + 1
        year_end = century * 100
        return ConstraintStrategyBuilder.temporal_year_range(
            time_property=time_property,
            year_start=year_start,
            year_end=year_end
        )

    @staticmethod
    def temporal_active_during(
        year: int,
        start_property: str = "P580",
        end_property: str = "P582",
        use_qualifier: bool = True,
        statement_var: str = "statement"
    ) -> Constraint:
        """
        Filter entities active during a specific year (start <= year <= end).
        Handles open-ended ranges (no end date = still active).
        """
        if use_qualifier:
            sparql_pattern = f"""
  OPTIONAL {{ ?{statement_var} pq:{start_property} ?startTime }}
  OPTIONAL {{ ?{statement_var} pq:{end_property} ?endTime }}
  FILTER(
    (!BOUND(?startTime) || YEAR(?startTime) <= {year}) &&
    (!BOUND(?endTime) || YEAR(?endTime) >= {year})
  )"""
        else:
            sparql_pattern = f"""
  OPTIONAL {{ ?entity wdt:{start_property} ?startTime }}
  OPTIONAL {{ ?entity wdt:{end_property} ?endTime }}
  FILTER(
    (!BOUND(?startTime) || YEAR(?startTime) <= {year}) &&
    (!BOUND(?endTime) || YEAR(?endTime) >= {year})
  )"""

        return Constraint(
            constraint_type=ConstraintType.TEMPORAL,
            property_id=f"{start_property}/{end_property}",
            label="active_during",
            sparql_pattern=sparql_pattern.strip(),
            description=f"Active in {year}",
            parameters={"year": year}
        )

    # =========================================================================
    # GEOGRAPHIC CONSTRAINTS
    # =========================================================================

    @staticmethod
    def geographic_country(country_qid: str, location_property: str = "P17") -> Constraint:
        """Filter entities by country."""
        sparql_pattern = f"?entity wdt:{location_property} wd:{country_qid} ."

        return Constraint(
            constraint_type=ConstraintType.GEOGRAPHIC,
            property_id=location_property,
            label="country",
            sparql_pattern=sparql_pattern,
            description=f"Located in {country_qid}",
            parameters={"country_qid": country_qid}
        )

    @staticmethod
    def geographic_continent(continent_qid: str) -> Constraint:
        """Filter entities by continent."""
        sparql_pattern = f"?entity wdt:P30 wd:{continent_qid} ."

        return Constraint(
            constraint_type=ConstraintType.GEOGRAPHIC,
            property_id="P30",
            label="continent",
            sparql_pattern=sparql_pattern,
            description=f"On continent {continent_qid}",
            parameters={"continent_qid": continent_qid}
        )

    @staticmethod
    def geographic_admin_region(
        region_qid: str,
        include_subregions: bool = True
    ) -> Constraint:
        """Filter by administrative region (state, province, etc.)."""
        if include_subregions:
            sparql_pattern = f"?entity wdt:P131+ wd:{region_qid} ."
        else:
            sparql_pattern = f"?entity wdt:P131 wd:{region_qid} ."

        return Constraint(
            constraint_type=ConstraintType.GEOGRAPHIC,
            property_id="P131",
            label="admin_region",
            sparql_pattern=sparql_pattern,
            description=f"In region {region_qid}",
            parameters={"region_qid": region_qid, "include_subregions": include_subregions}
        )

    @staticmethod
    def geographic_citizenship(country_qid: str) -> Constraint:
        """Filter people by country of citizenship."""
        sparql_pattern = f"?entity wdt:P27 wd:{country_qid} ."

        return Constraint(
            constraint_type=ConstraintType.GEOGRAPHIC,
            property_id="P27",
            label="citizenship",
            sparql_pattern=sparql_pattern,
            description=f"Citizen of {country_qid}",
            parameters={"country_qid": country_qid}
        )

    # =========================================================================
    # THRESHOLD CONSTRAINTS
    # =========================================================================

    @staticmethod
    def threshold_numeric(
        property_id: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        unit_qid: Optional[str] = None
    ) -> Constraint:
        """Filter by numeric property threshold."""
        sparql_pattern = f"?entity wdt:{property_id} ?value ."

        filters = []
        if min_value is not None:
            filters.append(f"?value >= {min_value}")
        if max_value is not None:
            filters.append(f"?value <= {max_value}")

        if filters:
            sparql_pattern += f"\n  FILTER({' && '.join(filters)})"

        description = f"Property {property_id}"
        if min_value is not None and max_value is not None:
            description = f"{property_id} between {min_value} and {max_value}"
        elif min_value is not None:
            description = f"{property_id} >= {min_value}"
        elif max_value is not None:
            description = f"{property_id} <= {max_value}"

        return Constraint(
            constraint_type=ConstraintType.THRESHOLD,
            property_id=property_id,
            label="numeric_threshold",
            sparql_pattern=sparql_pattern,
            description=description,
            parameters={"min_value": min_value, "max_value": max_value, "unit_qid": unit_qid}
        )

    @staticmethod
    def threshold_top_n_by_property(
        ranking_property: str,
        n: int,
        descending: bool = True
    ) -> Constraint:
        """
        Select top N entities by a ranking property.
        Note: This uses LIMIT and ORDER BY, applied at query level.
        """
        order_dir = "DESC" if descending else "ASC"
        sparql_pattern = f"""?entity wdt:{ranking_property} ?rankValue .
}}
ORDER BY {order_dir}(?rankValue)
LIMIT {n}"""

        return Constraint(
            constraint_type=ConstraintType.THRESHOLD,
            property_id=ranking_property,
            label="top_n",
            sparql_pattern=sparql_pattern,
            description=f"Top {n} by {ranking_property}",
            parameters={"n": n, "descending": descending}
        )

    @staticmethod
    def threshold_has_property(property_id: str) -> Constraint:
        """Filter to entities that have a specific property defined."""
        sparql_pattern = f"?entity wdt:{property_id} ?_{property_id}_value ."

        return Constraint(
            constraint_type=ConstraintType.THRESHOLD,
            property_id=property_id,
            label="has_property",
            sparql_pattern=sparql_pattern,
            description=f"Has {property_id} defined",
            parameters={"property_id": property_id}
        )

    # =========================================================================
    # CATEGORICAL CONSTRAINTS
    # =========================================================================

    @staticmethod
    def categorical_instance_of(type_qid: str, include_subclasses: bool = True) -> Constraint:
        """Filter by instance-of type."""
        if include_subclasses:
            sparql_pattern = f"?entity wdt:P31/wdt:P279* wd:{type_qid} ."
        else:
            sparql_pattern = f"?entity wdt:P31 wd:{type_qid} ."

        return Constraint(
            constraint_type=ConstraintType.CATEGORICAL,
            property_id="P31",
            label="instance_of",
            sparql_pattern=sparql_pattern,
            description=f"Instance of {type_qid}",
            parameters={"type_qid": type_qid, "include_subclasses": include_subclasses}
        )

    @staticmethod
    def categorical_occupation(occupation_qid: str) -> Constraint:
        """Filter people by occupation."""
        sparql_pattern = f"?entity wdt:P106 wd:{occupation_qid} ."

        return Constraint(
            constraint_type=ConstraintType.CATEGORICAL,
            property_id="P106",
            label="occupation",
            sparql_pattern=sparql_pattern,
            description=f"Occupation: {occupation_qid}",
            parameters={"occupation_qid": occupation_qid}
        )

    @staticmethod
    def categorical_field(field_qid: str) -> Constraint:
        """Filter by field of work."""
        sparql_pattern = f"?entity wdt:P101 wd:{field_qid} ."

        return Constraint(
            constraint_type=ConstraintType.CATEGORICAL,
            property_id="P101",
            label="field_of_work",
            sparql_pattern=sparql_pattern,
            description=f"Field: {field_qid}",
            parameters={"field_qid": field_qid}
        )

    @staticmethod
    def categorical_gender(gender_qid: str) -> Constraint:
        """Filter people by gender (Q6581097=male, Q6581072=female)."""
        sparql_pattern = f"?entity wdt:P21 wd:{gender_qid} ."

        return Constraint(
            constraint_type=ConstraintType.CATEGORICAL,
            property_id="P21",
            label="gender",
            sparql_pattern=sparql_pattern,
            description=f"Gender: {gender_qid}",
            parameters={"gender_qid": gender_qid}
        )

    @staticmethod
    def categorical_award_category(
        award_qid: str,
        category_qid: str
    ) -> Constraint:
        """Filter award recipients by award category."""
        sparql_pattern = f"""?entity p:P166 ?awardStatement .
  ?awardStatement ps:P166 wd:{award_qid} .
  ?awardStatement pq:P2868 wd:{category_qid} ."""

        return Constraint(
            constraint_type=ConstraintType.CATEGORICAL,
            property_id="P166",
            label="award_category",
            sparql_pattern=sparql_pattern,
            description=f"Award {award_qid} category {category_qid}",
            parameters={"award_qid": award_qid, "category_qid": category_qid}
        )

    # =========================================================================
    # COMPOSITE CONSTRAINT BUILDER
    # =========================================================================

    @staticmethod
    def combine_constraints(constraints: list, operator: str = "AND") -> Constraint:
        """
        Combine multiple constraints into a composite constraint.

        Args:
            constraints: List of Constraint objects to combine
            operator: "AND" (all must match) or "OR" (any must match)
        """
        if operator == "AND":
            # AND: just concatenate the patterns
            combined_pattern = "\n  ".join(c.sparql_pattern for c in constraints)
            description = " AND ".join(c.description for c in constraints)
        else:
            # OR: use UNION
            union_blocks = [f"{{ {c.sparql_pattern} }}" for c in constraints]
            combined_pattern = " UNION ".join(union_blocks)
            description = " OR ".join(c.description for c in constraints)

        return Constraint(
            constraint_type=ConstraintType.COMPOSITE,
            property_id="composite",
            label=f"composite_{operator.lower()}",
            sparql_pattern=combined_pattern,
            description=description,
            parameters={"operator": operator, "constraints": [c.label for c in constraints]}
        )


# =============================================================================
# PREDEFINED CONSTRAINT SETS
# =============================================================================

# Geographic regions
CONTINENTS = {
    "europe": "Q46",
    "asia": "Q48",
    "africa": "Q15",
    "north_america": "Q49",
    "south_america": "Q18",
    "oceania": "Q55643"
}

MAJOR_COUNTRIES = {
    "united_states": "Q30",
    "united_kingdom": "Q145",
    "france": "Q142",
    "germany": "Q183",
    "japan": "Q17",
    "china": "Q148",
    "india": "Q668",
    "brazil": "Q155",
    "russia": "Q159",
    "canada": "Q16"
}

# Common entity types
ENTITY_TYPES = {
    "human": "Q5",
    "country": "Q6256",
    "city": "Q515",
    "film": "Q11424",
    "company": "Q4830453",
    "university": "Q3918",
    "book": "Q7725634",
    "album": "Q482994",
    "tv_series": "Q5398426"
}

# Occupations
OCCUPATIONS = {
    "physicist": "Q169470",
    "chemist": "Q593644",
    "biologist": "Q864503",
    "mathematician": "Q170790",
    "actor": "Q33999",
    "director": "Q2526255",
    "writer": "Q36180",
    "musician": "Q639669",
    "politician": "Q82955",
    "astronaut": "Q11631"
}

# Genders
GENDERS = {
    "male": "Q6581097",
    "female": "Q6581072"
}


class ConstraintPresets:
    """Preset constraint combinations for common use cases."""

    @staticmethod
    def female_scientists_by_country(country_qid: str, field_qid: str) -> list:
        """Women scientists in a specific field from a specific country."""
        builder = ConstraintStrategyBuilder
        return [
            builder.categorical_instance_of(ENTITY_TYPES["human"]),
            builder.categorical_gender(GENDERS["female"]),
            builder.categorical_field(field_qid),
            builder.geographic_citizenship(country_qid)
        ]

    @staticmethod
    def award_winners_by_decade(award_qid: str, decade: int) -> list:
        """Award winners from a specific decade."""
        builder = ConstraintStrategyBuilder
        return [
            builder.temporal_decade(decade, use_qualifier=True)
        ]

    @staticmethod
    def companies_in_region_by_size(region_qid: str, min_employees: int) -> list:
        """Companies in a region above employee threshold."""
        builder = ConstraintStrategyBuilder
        return [
            builder.categorical_instance_of(ENTITY_TYPES["company"]),
            builder.geographic_admin_region(region_qid),
            builder.threshold_numeric("P1128", min_value=min_employees)  # P1128 = employees
        ]

    @staticmethod
    def historical_figures_by_century(century: int, occupation_qid: str) -> list:
        """Notable people from a specific century and occupation."""
        builder = ConstraintStrategyBuilder
        return [
            builder.categorical_instance_of(ENTITY_TYPES["human"]),
            builder.categorical_occupation(occupation_qid),
            builder.temporal_century(century, time_property="P569")  # P569 = birth date
        ]


def apply_constraints_to_query(base_sparql: str, constraints: list) -> str:
    """
    Apply a list of constraints to a base SPARQL query.

    Args:
        base_sparql: Base SPARQL query with WHERE clause
        constraints: List of Constraint objects to apply

    Returns:
        Modified SPARQL query with constraints applied
    """
    # Find the WHERE clause
    if "WHERE" not in base_sparql.upper():
        raise ValueError("Base query must contain a WHERE clause")

    # Insert constraints before the closing brace of WHERE
    constraint_patterns = "\n  ".join(c.sparql_pattern for c in constraints)

    # Find the last closing brace that ends the WHERE clause
    # This is a simplified approach - real implementation would parse the query
    parts = base_sparql.rsplit("}", 1)
    if len(parts) == 2:
        modified_query = f"{parts[0]}\n  {constraint_patterns}\n}}{parts[1]}"
    else:
        modified_query = base_sparql.rstrip("}") + f"\n  {constraint_patterns}\n}}"

    return modified_query


if __name__ == "__main__":
    # Example usage
    builder = ConstraintStrategyBuilder

    print("=== Temporal Constraints ===")
    c1 = builder.temporal_year_range(year_start=2010, year_end=2020)
    print(f"{c1.description}:\n{c1.sparql_pattern}\n")

    c2 = builder.temporal_decade(1990)
    print(f"{c2.description}:\n{c2.sparql_pattern}\n")

    print("=== Geographic Constraints ===")
    c3 = builder.geographic_country(MAJOR_COUNTRIES["japan"])
    print(f"{c3.description}:\n{c3.sparql_pattern}\n")

    c4 = builder.geographic_continent(CONTINENTS["europe"])
    print(f"{c4.description}:\n{c4.sparql_pattern}\n")

    print("=== Categorical Constraints ===")
    c5 = builder.categorical_occupation(OCCUPATIONS["physicist"])
    print(f"{c5.description}:\n{c5.sparql_pattern}\n")

    c6 = builder.categorical_gender(GENDERS["female"])
    print(f"{c6.description}:\n{c6.sparql_pattern}\n")

    print("=== Composite Constraint ===")
    composite = builder.combine_constraints([c5, c6, c3], operator="AND")
    print(f"{composite.description}:\n{composite.sparql_pattern}\n")

    print("=== Preset: Female Japanese Physicists ===")
    preset_constraints = ConstraintPresets.female_scientists_by_country(
        MAJOR_COUNTRIES["japan"],
        "Q413"  # physics
    )
    for c in preset_constraints:
        print(f"  - {c.description}")
