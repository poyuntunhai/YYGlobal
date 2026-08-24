from app.services.rankings import (
    qs_2026_top300,
    qs_rank_for_university,
    ranked_university_options,
    same_university,
)


def test_qs_2026_dataset_contains_exactly_top_300_universities():
    rows = qs_2026_top300()
    assert len(rows) == 300
    assert min(int(item["rank"]) for item in rows) == 1
    assert max(int(item["rank"]) for item in rows) == 300
    assert qs_rank_for_university("Massachusetts Institute of Technology") == 1
    assert qs_rank_for_university("Nanyang Technological University") == 12
    assert same_university(
        "National University of Singapore",
        "National University of Singapore (NUS)",
    )
    assert same_university("MIT", "Massachusetts Institute of Technology")


def test_ranked_university_options_apply_country_and_rank_together():
    options = ranked_university_options(["Hong Kong", "Singapore"], 50)
    names = {str(item["university"]) for item in options}
    assert names == {
        "National University of Singapore (NUS)",
        "Nanyang Technological University",
        "The University of Hong Kong",
        "The Chinese University of Hong Kong",
        "The Hong Kong University of Science and Technology",
    }
    assert all(int(item["qs_rank"]) <= 50 for item in options)
    assert all(item["country"] in {"Hong Kong", "Singapore"} for item in options)
