"""add recommendation goals and QS ranking metadata

Revision ID: 0016_recommendation_goals_qs
Revises: 0015_conversation_memory_state
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0016_recommendation_goals_qs"
down_revision: Union[str, None] = "0015_conversation_memory_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    program_columns = {item["name"] for item in inspector.get_columns("programs")}
    with op.batch_alter_table("programs") as batch_op:
        if "qs_rank" not in program_columns:
            batch_op.add_column(sa.Column("qs_rank", sa.Integer(), nullable=True))
            batch_op.create_index("ix_programs_qs_rank", ["qs_rank"])
        if "qs_ranking_year" not in program_columns:
            batch_op.add_column(sa.Column("qs_ranking_year", sa.Integer(), nullable=True))

    inspector = sa.inspect(op.get_bind())
    if "recommendation_goals" not in inspector.get_table_names():
        op.create_table(
            "recommendation_goals",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("owner_id", sa.String(64), nullable=False),
            sa.Column("target_countries", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("target_fields", sa.JSON(), nullable=False, server_default="[]"),
            sa.Column("intake", sa.String(80), nullable=False, server_default=""),
            sa.Column("budget", sa.Float(), nullable=True),
            sa.Column("max_qs_rank", sa.Integer(), nullable=True, server_default="100"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("owner_id", name="uq_recommendation_goals_owner"),
        )
        op.create_index(
            "ix_recommendation_goals_owner_id", "recommendation_goals", ["owner_id"]
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "recommendation_goals" in inspector.get_table_names():
        op.drop_table("recommendation_goals")
    program_columns = {item["name"] for item in inspector.get_columns("programs")}
    with op.batch_alter_table("programs") as batch_op:
        if "qs_ranking_year" in program_columns:
            batch_op.drop_column("qs_ranking_year")
        if "qs_rank" in program_columns:
            batch_op.drop_index("ix_programs_qs_rank")
            batch_op.drop_column("qs_rank")
