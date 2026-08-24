"""add target degree level to recommendation goals

Revision ID: 0018_goal_degree
Revises: 0017_goal_university
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0018_goal_degree"
down_revision: Union[str, None] = "0017_goal_university"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {item["name"] for item in inspector.get_columns("recommendation_goals")}
    if "target_degree_level" not in columns:
        with op.batch_alter_table("recommendation_goals") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "target_degree_level",
                    sa.String(30),
                    nullable=False,
                    server_default="master",
                )
            )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {item["name"] for item in inspector.get_columns("recommendation_goals")}
    if "target_degree_level" in columns:
        with op.batch_alter_table("recommendation_goals") as batch_op:
            batch_op.drop_column("target_degree_level")
