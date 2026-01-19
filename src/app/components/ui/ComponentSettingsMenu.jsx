import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiDotsVertical } from 'react-icons/hi';
import './ComponentSettingsMenu.scss';

export function ComponentSettingsMenu({
  visualizationKey,
  currentLabels = {},
  onUpdateLabels
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [editingField, setEditingField] = useState(null);
  const [tempValue, setTempValue] = useState('');

  // Define all possible fields - show only the ones that are defined in currentLabels
  const allFields = [
    { key: 'title', label: 'Title' },
    { key: 'xAxisLabel', label: 'X-Axis Label' },
    { key: 'yAxisLabel', label: 'Y-Axis Label' }
  ];

  // Only show fields that exist in the initial currentLabels object
  const fields = allFields.filter(field => field.key in currentLabels);

  const handleEdit = (fieldKey, currentValue) => {
    setEditingField(fieldKey);
    setTempValue(currentValue || '');
  };

  const handleChange = (e) => {
    const newValue = e.target.value;
    setTempValue(newValue);
    // Update on every keystroke
    if (editingField) {
      onUpdateLabels(visualizationKey, editingField, newValue);
    }
  };

  const handleBlur = () => {
    setEditingField(null);
    setTempValue('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === 'Escape') {
      e.target.blur(); // Exit edit mode
    }
  };

  if (fields.length === 0) return null;

  return (
    <div className="viz-component-settings">
      <motion.button
        className="viz-settings-icon-btn"
        onClick={() => setIsOpen(!isOpen)}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        title="Edit labels"
      >
        <HiDotsVertical />
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              className="viz-settings-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => {
                setIsOpen(false);
                setEditingField(null);
              }}
            />
            <motion.div
              className="viz-settings-dropdown"
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              transition={{ type: 'spring', damping: 25, stiffness: 400 }}
            >
              <div className="viz-settings-header">Edit Labels</div>
              <div className="viz-settings-fields">
                {fields.map(field => (
                  <div key={field.key} className="viz-settings-field">
                    <label>{field.label}</label>
                    {editingField === field.key ? (
                      <input
                        type="text"
                        className="viz-settings-input"
                        value={tempValue}
                        onChange={handleChange}
                        onKeyDown={handleKeyDown}
                        onBlur={handleBlur}
                        autoFocus
                        placeholder={field.label}
                      />
                    ) : (
                      <div
                        className="viz-settings-value-row"
                        onClick={() => handleEdit(field.key, currentLabels[field.key])}
                      >
                        <span className="viz-settings-current-value">
                          {currentLabels[field.key] || '(none)'}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
