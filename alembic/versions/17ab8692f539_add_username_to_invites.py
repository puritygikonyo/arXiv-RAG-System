"""Add username to invites

Revision ID: 17ab8692f539
Revises: dc22e2b91c3b
Create Date: 2026-07-24 13:31:38.851443
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '17ab8692f539'
down_revision: Union[str, Sequence[str], None] = 'dc22e2b91c3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add as nullable first — a NOT NULL column with no default would
    #    fail immediately against the existing invite row(s) already in
    #    this table.
    op.add_column('invites', sa.Column('username', sa.String(length=64), nullable=True))

    # 2. Backfill existing rows with a placeholder derived from their id,
    #    so they get a valid, unique value instead of staying NULL.
    #    Rename these manually afterward if you want a nicer username for
    #    any pre-existing invite (check with list_invites.py).
    op.execute(
        "UPDATE invites SET username = 'user_' || id::text WHERE username IS NULL"
    )

    # 3. Now that every row has a value, enforce NOT NULL and add the
    #    unique index, matching the model definition.
    op.alter_column('invites', 'username', nullable=False)
    op.create_index(op.f('ix_invites_username'), 'invites', ['username'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_invites_username'), table_name='invites')
    op.drop_column('invites', 'username')