<svg viewBox="0 0 800 1400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#4F46E5;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#059669;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#10B981;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad3" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#DC2626;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#EF4444;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- Title -->
  <text x="400" y="30" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#1F2937" text-anchor="middle">
    Traffic Sign Recognition System
  </text>
  
  <!-- Step 1: Database -->
  <rect x="150" y="60" width="500" height="80" rx="10" fill="url(#grad1)" filter="url(#shadow)"/>
  <text x="400" y="90" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    1. Traffic Signs Database
  </text>
  <text x="400" y="115" font-family="Arial, sans-serif" font-size="13" fill="white" text-anchor="middle">
    Curated dataset (GTSRB or custom symbols)
  </text>
  <text x="400" y="132" font-family="Arial, sans-serif" font-size="13" fill="white" text-anchor="middle">
    with labeled traffic sign images
  </text>
  
  <!-- Arrow -->
  <path d="M 400 140 L 400 160" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 2: Preprocessing -->
  <rect x="150" y="160" width="500" height="100" rx="10" fill="url(#grad2)" filter="url(#shadow)"/>
  <text x="400" y="185" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    2. Data Preprocessing
  </text>
  <text x="400" y="207" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Resize (32x32) • Orientation correction
  </text>
  <text x="400" y="225" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Grayscale conversion • Data augmentation
  </text>
  <text x="400" y="243" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    (zoom, shifts, rotation)
  </text>
  
  <!-- Arrow -->
  <path d="M 400 260 L 400 280" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 3: Model Architecture -->
  <rect x="150" y="280" width="500" height="100" rx="10" fill="url(#grad1)" filter="url(#shadow)"/>
  <text x="400" y="305" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    3. Dynamic Model Architecture
  </text>
  <text x="400" y="327" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    CNN models (LeNet/ResNet variants)
  </text>
  <text x="400" y="345" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Tunable depths • Batch normalization
  </text>
  <text x="400" y="363" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Dropout regularization • Learning rate tuning
  </text>
  
  <!-- Arrow -->
  <path d="M 400 380 L 400 400" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 4: Grid Search -->
  <rect x="150" y="400" width="500" height="100" rx="10" fill="url(#grad3)" filter="url(#shadow)"/>
  <text x="400" y="425" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    4. Hyperparameters Grid Search
  </text>
  <text x="400" y="447" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    Grid search over combinations:
  </text>
  <text x="400" y="465" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    Depths [3,5,7] • LR [0.001,0.01]
  </text>
  <text x="400" y="483" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    Dropout [0.2,0.5] • Train/validation splits
  </text>
  
  <!-- Arrow -->
  <path d="M 400 500 L 400 520" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 5: Validation -->
  <rect x="150" y="520" width="500" height="80" rx="10" fill="url(#grad2)" filter="url(#shadow)"/>
  <text x="400" y="550" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    5. Validate Each Model
  </text>
  <text x="400" y="575" font-family="Arial, sans-serif" font-size="13" fill="white" text-anchor="middle">
    K-fold cross-validation for loss & accuracy
  </text>
  <text x="400" y="592" font-family="Arial, sans-serif" font-size="13" fill="white" text-anchor="middle">
    on held-out validation data
  </text>
  
  <!-- Arrow -->
  <path d="M 400 600 L 400 620" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 6: Select Best Model -->
  <rect x="150" y="620" width="500" height="70" rx="10" fill="url(#grad1)" filter="url(#shadow)"/>
  <text x="400" y="650" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    6. Select Best Model
  </text>
  <text x="400" y="675" font-family="Arial, sans-serif" font-size="13" fill="white" text-anchor="middle">
    Choose top performer (highest F1-score)
  </text>
  
  <!-- Arrow -->
  <path d="M 400 690 L 400 710" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 7: Testing -->
  <rect x="150" y="710" width="500" height="70" rx="10" fill="url(#grad3)" filter="url(#shadow)"/>
  <text x="400" y="740" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    7. Testing
  </text>
  <text x="400" y="765" font-family="Arial, sans-serif" font-size="13" fill="white" text-anchor="middle">
    Evaluate on unseen test set for generalization
  </text>
  
  <!-- Arrow -->
  <path d="M 400 780 L 400 800" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 8: Evaluation -->
  <rect x="150" y="800" width="500" height="100" rx="10" fill="url(#grad2)" filter="url(#shadow)"/>
  <text x="400" y="825" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    8. Evaluation
  </text>
  <text x="400" y="847" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Confusion matrix • Overall accuracy
  </text>
  <text x="400" y="865" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Grad-CAM heatmaps for interpretability
  </text>
  <text x="400" y="883" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    (visualizing attention on sign features)
  </text>
  
  <!-- Arrow -->
  <path d="M 400 900 L 400 920" stroke="#6B7280" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Step 9: Deployment -->
  <rect x="150" y="920" width="500" height="100" rx="10" fill="url(#grad1)" filter="url(#shadow)"/>
  <text x="400" y="945" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">
    9. Deployment
  </text>
  <text x="400" y="967" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    Streamlit app for real-time inference
  </text>
  <text x="400" y="985" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • Users upload images • Get predictions
  </text>
  <text x="400" y="1003" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle">
    • View overlaid Grad-CAM heatmaps
  </text>
  
  <!-- Arrow markers -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
      <polygon points="0 0, 10 5, 0 10" fill="#6B7280" />
    </marker>
  </defs>
  
  <!-- Icons/Decorations -->
  <g transform="translate(80, 90)">
    <circle cx="20" cy="0" r="18" fill="#FEF3C7" stroke="#F59E0B" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#F59E0B" text-anchor="middle">📊</text>
  </g>
  
  <g transform="translate(80, 190)">
    <circle cx="20" cy="0" r="18" fill="#D1FAE5" stroke="#059669" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#059669" text-anchor="middle">⚙️</text>
  </g>
  
  <g transform="translate(80, 310)">
    <circle cx="20" cy="0" r="18" fill="#E0E7FF" stroke="#4F46E5" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#4F46E5" text-anchor="middle">🧠</text>
  </g>
  
  <g transform="translate(80, 430)">
    <circle cx="20" cy="0" r="18" fill="#FECACA" stroke="#DC2626" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#DC2626" text-anchor="middle">🔍</text>
  </g>
  
  <g transform="translate(80, 550)">
    <circle cx="20" cy="0" r="18" fill="#D1FAE5" stroke="#059669" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#059669" text-anchor="middle">✓</text>
  </g>
  
  <g transform="translate(80, 645)">
    <circle cx="20" cy="0" r="18" fill="#E0E7FF" stroke="#4F46E5" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#4F46E5" text-anchor="middle">🏆</text>
  </g>
  
  <g transform="translate(80, 735)">
    <circle cx="20" cy="0" r="18" fill="#FECACA" stroke="#DC2626" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#DC2626" text-anchor="middle">🧪</text>
  </g>
  
  <g transform="translate(80, 835)">
    <circle cx="20" cy="0" r="18" fill="#D1FAE5" stroke="#059669" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#059669" text-anchor="middle">📈</text>
  </g>
  
  <g transform="translate(80, 955)">
    <circle cx="20" cy="0" r="18" fill="#E0E7FF" stroke="#4F46E5" stroke-width="2"/>
    <text x="20" y="6" font-family="Arial, sans-serif" font-size="20" fill="#4F46E5" text-anchor="middle">🚀</text>
  </g>
</svg>