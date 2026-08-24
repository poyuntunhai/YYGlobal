"""add official university catalog URL to programs

Revision ID: 0019_program_catalog_url
Revises: 0018_goal_degree
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0019_program_catalog_url"
down_revision: Union[str, None] = "0018_goal_degree"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("programs")}
    if "catalog_url" not in columns:
        with op.batch_alter_table("programs") as batch_op:
            batch_op.add_column(
                sa.Column("catalog_url", sa.String(500), nullable=False, server_default="")
            )


def downgrade() -> None:
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("programs")}
    if "catalog_url" in columns:
        with op.batch_alter_table("programs") as batch_op:
            batch_op.drop_column("catalog_url")
