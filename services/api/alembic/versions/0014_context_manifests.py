"""add material writing context manifests

Revision ID: 0014_context_manifests
Revises: 0013_schema_alignment
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0014_context_manifests"
down_revision: Union[str, None] = "0013_schema_alignment"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "context_manifests" not in inspector.get_table_names():
        op.create_table(
            "context_manifests",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("agent_run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=True),
            sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=True),
            sa.Column("program_id", sa.String(36), sa.ForeignKey("programs.id"), nullable=True),
            sa.Column("material_kind", sa.String(40), nullable=False, server_default=""),
            sa.Column("slot_key", sa.String(120), nullable=False, server_default=""),
            sa.Column("intent", sa.String(30), nullable=False, server_default="chat"),
            sa.Column("profile_ref", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("conversation_ref", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("experience_refs", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("document_refs", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("draft_refs", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("official_evidence_refs", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("current_draft_ref", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("content_hashes", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("completeness", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("owner_id", sa.String(64), nullable=False, server_default="local-admin"),
        )
        for column in (
            "agent_run_id", "conversation_id", "program_id", "material_kind",
            "slot_key", "intent", "status", "owner_id",
        ):
            op.create_index(
                f"ix_context_manifests_{column}", "context_manifests", [column]
            )
    columns = {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns("material_drafts")
    }
    with op.batch_alter_table("material_drafts") as batch_op:
        if "context_manifest_id" not in columns:
            batch_op.add_column(sa.Column("context_manifest_id", sa.String(36), nullable=True))
            batch_op.create_foreign_key(
                "fk_material_drafts_context_manifest_id_context_manifests",
                "context_manifests",
                ["context_manifest_id"],
                ["id"],
            )
            batch_op.create_index(
                "ix_material_drafts_context_manifest_id", ["context_manifest_id"]
            )
        if "source_refs" not in columns:
            batch_op.add_column(
                sa.Column("source_refs", sa.JSON(), nullable=False, server_default="{}")
            )


def downgrade() -> None:
    columns = {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns("material_drafts")
    }
    with op.batch_alter_table("material_drafts") as batch_op:
        if "source_refs" in columns:
            batch_op.drop_column("source_refs")
        if "context_manifest_id" in columns:
            batch_op.drop_index("ix_material_drafts_context_manifest_id")
            batch_op.drop_constraint(
                "fk_material_drafts_context_manifest_id_context_manifests",
                type_="foreignkey",
            )
            batch_op.drop_column("context_manifest_id")
    if "context_manifests" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("context_manifests")
