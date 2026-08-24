"""align package, conversation, and draft constraints

Revision ID: 0013_schema_alignment
Revises: 0012_material_draft_lineage
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0013_schema_alignment"
down_revision: Union[str, None] = "0012_material_draft_lineage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_index(table: str, name: str, columns: list[str]) -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in indexes:
        op.create_index(name, table, columns)


def _has_foreign_key(table: str, columns: list[str], referred_table: str) -> bool:
    return any(
        item.get("constrained_columns") == columns
        and item.get("referred_table") == referred_table
        for item in sa.inspect(op.get_bind()).get_foreign_keys(table)
    )


def upgrade() -> None:
    _create_index(
        "application_packages",
        "ix_application_packages_plan_confirmed",
        ["plan_confirmed"],
    )
    for column in ("material_kind", "pinned", "program_id", "slot_key"):
        _create_index("conversations", f"ix_conversations_{column}", [column])
    _create_index(
        "material_drafts", "ix_material_drafts_derived_from_id", ["derived_from_id"]
    )
    _create_index("material_drafts", "ix_material_drafts_slot_key", ["slot_key"])

    if not _has_foreign_key("conversations", ["program_id"], "programs"):
        with op.batch_alter_table("conversations") as batch_op:
            batch_op.create_foreign_key(
                "fk_conversations_program_id_programs",
                "programs",
                ["program_id"],
                ["id"],
            )
    if not _has_foreign_key(
        "material_drafts", ["derived_from_id"], "material_drafts"
    ):
        with op.batch_alter_table("material_drafts") as batch_op:
            batch_op.create_foreign_key(
                "fk_material_drafts_derived_from_id_material_drafts",
                "material_drafts",
                ["derived_from_id"],
                ["id"],
            )


def downgrade() -> None:
    foreign_keys = sa.inspect(op.get_bind()).get_foreign_keys("material_drafts")
    if any(
        item.get("name") == "fk_material_drafts_derived_from_id_material_drafts"
        for item in foreign_keys
    ):
        with op.batch_alter_table("material_drafts") as batch_op:
            batch_op.drop_constraint(
                "fk_material_drafts_derived_from_id_material_drafts",
                type_="foreignkey",
            )
    foreign_keys = sa.inspect(op.get_bind()).get_foreign_keys("conversations")
    if any(
        item.get("name") == "fk_conversations_program_id_programs"
        for item in foreign_keys
    ):
        with op.batch_alter_table("conversations") as batch_op:
            batch_op.drop_constraint(
                "fk_conversations_program_id_programs", type_="foreignkey"
            )

    for table, names in (
        (
            "material_drafts",
            ["ix_material_drafts_slot_key", "ix_material_drafts_derived_from_id"],
        ),
        (
            "conversations",
            [
                "ix_conversations_slot_key",
                "ix_conversations_program_id",
                "ix_conversations_pinned",
                "ix_conversations_material_kind",
            ],
        ),
        ("application_packages", ["ix_application_packages_plan_confirmed"]),
    ):
        existing = {
            item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)
        }
        for name in names:
            if name in existing:
                op.drop_index(name, table_name=table)
