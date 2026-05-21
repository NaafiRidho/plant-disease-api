"""create detection_histories table

Revision ID: b1c2d3e4f5a6
Revises: 471ae54163b3
Create Date: 2026-05-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b1c2d3e4f5a6'
down_revision = '471ae54163b3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'detection_histories',
        sa.Column('id',              sa.Integer(),      nullable=False),
        sa.Column('filename',        sa.String(255),    nullable=True),
        sa.Column('predicted_class', sa.String(100),    nullable=False),
        sa.Column('confidence',      sa.Float(),        nullable=False),
        sa.Column('plant_type',      sa.String(50),     nullable=True),
        sa.Column('disease_name',    sa.String(100),    nullable=True),
        sa.Column('is_healthy',      sa.Boolean(),      nullable=False, server_default=sa.false()),
        sa.Column('ip_address',      sa.String(50),     nullable=True),
        sa.Column('top_3_json',      sa.Text(),         nullable=True),
        sa.Column('user_id',         sa.Integer(),      nullable=True),
        sa.Column('created_at',      sa.DateTime(),     nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Indexes untuk query yang sering dipakai
    op.create_index('ix_dh_plant_type',  'detection_histories', ['plant_type'])
    op.create_index('ix_dh_is_healthy',  'detection_histories', ['is_healthy'])
    op.create_index('ix_dh_created_at',  'detection_histories', ['created_at'])
    op.create_index('ix_dh_user_id',     'detection_histories', ['user_id'])


def downgrade():
    op.drop_index('ix_dh_user_id',    table_name='detection_histories')
    op.drop_index('ix_dh_created_at', table_name='detection_histories')
    op.drop_index('ix_dh_is_healthy', table_name='detection_histories')
    op.drop_index('ix_dh_plant_type', table_name='detection_histories')
    op.drop_table('detection_histories')
