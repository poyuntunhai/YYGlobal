"""add persistent conversation memory state

Revision ID: 0015_conversation_memory_state
Revises: 0014_context_manifests
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0015_conversation_memory_state"
down_revision: Union[str, None] = "0014_context_manifests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns("conversations")
    }
    if "memory_state" not in columns:
        with op.batch_alter_table("conversations") as batch_op:
            batch_op.add_column(
                sa.Column("memory_state", sa.JSON(), nullable=False, server_default="{}")
            )


def downgrade() -> None:
    columns = {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns("conversations")
    }
    if "memory_state" in columns:
        with op.batch_alter_table("conversations") as batch_op:
            batch_op.drop_column("memory_state")
