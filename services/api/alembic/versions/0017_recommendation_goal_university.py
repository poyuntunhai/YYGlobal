"""add target university to recommendation goals

Revision ID: 0017_goal_university
Revises: 0016_recommendation_goals_qs
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0017_goal_university"
down_revision: Union[str, None] = "0016_recommendation_goals_qs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {item["name"] for item in inspector.get_columns("recommendation_goals")}
    if "target_university" not in columns:
        with op.batch_alter_table("recommendation_goals") as batch_op:
            batch_op.add_column(
                sa.Column("target_university", sa.String(200), nullable=False, server_default="")
            )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {item["name"] for item in inspector.get_columns("recommendation_goals")}
    if "target_university" in columns:
        with op.batch_alter_table("recommendation_goals") as batch_op:
            batch_op.drop_column("target_university")
