import type { PartVisibility } from '../types';

const PART_LABELS: Record<keyof PartVisibility, string> = {
  leftEye:   'Left Eye',
  rightEye:  'Right Eye',
  leftBrow:  'Left Brow',
  rightBrow: 'Right Brow',
  nose:      'Nose',
  lips:      'Lips',
  skin:      'Skin',
};

const PART_COLORS: Record<keyof PartVisibility, string> = {
  leftEye:   '#64b4ff',
  rightEye:  '#64b4ff',
  leftBrow:  '#ffc850',
  rightBrow: '#ffc850',
  nose:      '#8cff8c',
  lips:      '#ff6478',
  skin:      '#c8a078',
};

interface Props {
  visibility: PartVisibility;
  onChange: (next: PartVisibility) => void;
  disabled?: boolean;
}

export function PartToggles({ visibility, onChange, disabled }: Props) {
  const keys = Object.keys(visibility) as (keyof PartVisibility)[];

  const toggle = (key: keyof PartVisibility) => {
    onChange({ ...visibility, [key]: !visibility[key] });
  };

  const allOn = keys.every(k => visibility[k]);
  const toggleAll = () => {
    const next = keys.reduce((acc, k) => ({ ...acc, [k]: !allOn }), {} as PartVisibility);
    onChange(next);
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <span style={styles.label}>Parts</span>
        <button
          style={{ ...styles.allBtn, opacity: disabled ? 0.4 : 1 }}
          disabled={disabled}
          onClick={toggleAll}
        >
          {allOn ? 'Hide all' : 'Show all'}
        </button>
      </div>
      <div style={styles.grid}>
        {keys.map(key => (
          <button
            key={key}
            style={{
              ...styles.chip,
              opacity: disabled ? 0.35 : 1,
              borderColor: visibility[key] ? PART_COLORS[key] : '#2a2a3a',
              background: visibility[key] ? `${PART_COLORS[key]}18` : 'transparent',
              color: visibility[key] ? PART_COLORS[key] : '#444',
            }}
            disabled={disabled}
            onClick={() => toggle(key)}
          >
            <span
              style={{
                ...styles.dot,
                background: PART_COLORS[key],
                opacity: visibility[key] ? 1 : 0.3,
              }}
            />
            {PART_LABELS[key]}
          </button>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  label: {
    fontSize: 13,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: '#666',
  },
  allBtn: {
    background: 'none',
    border: '1px solid #2a2a3a',
    borderRadius: 6,
    color: '#555',
    fontSize: 11,
    padding: '2px 8px',
    cursor: 'pointer',
  },
  grid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
  },
  chip: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    padding: '5px 10px',
    borderRadius: 20,
    border: '1px solid',
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 500,
    transition: 'all 0.15s',
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    display: 'inline-block',
    flexShrink: 0,
  },
};
