"""add scientific_name and severity to detection_histories

Revision ID: c3d4e5f6a7b8
Revises: 494284da7838
Create Date: 2026-06-09 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'c3d4e5f6a7b8'
down_revision = '494284da7838'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('detection_histories', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('scientific_name', sa.String(length=100), nullable=True)
        )
        batch_op.add_column(
            sa.Column('severity', sa.String(length=30), nullable=True)
        )


def downgrade():
    with op.batch_alter_table('detection_histories', schema=None) as batch_op:
        batch_op.drop_column('severity')
        batch_op.drop_column('scientific_name')
