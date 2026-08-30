# Waste-Weight-10K: Dataset Specification & Provenance

## 1. Overview & Key Statistics

The **Waste-Weight-10K** dataset is a multimodal benchmark designed for contactless physical weight estimation of industrial and commercial scrap objects from paired RGB imagery and spatial metadata.

| Metric | Value | Description |
|---|---|---|
| **Total Samples** | 10,421 | Paired RGB images with tabular metadata |
| **Material Categories** | 11 | Industrial and recyclable scrap classes |
| **Weight Range** | 3.5 kg – 3,450.0 kg | Wide dynamic mass range (filtered $\ge 50\text{ kg}$ in primary benchmark) |
| **Physical Setups** | 75 | Distinct object instances captured across multiple angles/distances |
| **Image Resolution** | $224 \times 224 \times 3$ | Standardized RGB inputs for Vision Transformer |
| **Ground Truth Sensor** | Industrial Load Cell | Certified precision $\pm 0.05\text{ kg}$ |

---

## 2. Spatial Measurement Conventions & Units

> [!IMPORTANT]
> All raw dimension and spatial position columns in `image.csv` are recorded in **centimeters ($\text{cm}$)**, and weight is recorded in **kilograms ($\text{kg}$)**.

| Variable | Column Name | Unit | Description | Range (Raw) |
|---|---|---|---|---|
| Width | `V_x` ($L_x$) | $\text{cm}$ | Object bounding width along the X-axis | $10.0 - 250.0\text{ cm}$ |
| Height | `V_y` ($L_y$) | $\text{cm}$ | Object bounding height along the Y-axis | $10.0 - 220.0\text{ cm}$ |
| Depth | `V_z` ($L_z$) | $\text{cm}$ | Object bounding depth along the Z-axis | $10.0 - 250.0\text{ cm}$ |
| Camera Distance | `D_x` ($D_x$) | $\text{cm}$ | Horizontal distance from camera optical center to object center | $50.0 - 450.0\text{ cm}$ |
| Camera Height | `D_y` ($D_y$) | $\text{cm}$ | Height of camera optical center above the ground plane | $30.0 - 260.0\text{ cm}$ |
| Target Weight | `weight_in_kg` | $\text{kg}$ | Calibrated ground-truth mass from load cell | $3.5 - 3,450.0\text{ kg}$ |
| Material Class | `Type` | string | Categorical material / scrap type | 11 distinct classes |

### Volume & Density Formulas
- **Raw Volume (Code Level)**:
  $$V_{\text{code}} = V_x \times V_y \times V_z \quad (\text{cm}^3)$$
- **SI Volume (Paper Reporting)**:
  $$V_{\text{SI}} = \frac{V_x \times V_y \times V_z}{10^6} \quad (\text{m}^3)$$
- **Effective Apparent Volume**:
  $$V_{\text{apparent}} = \frac{V_{\text{code}}}{D_x^2 + \epsilon}$$

---

## 3. Category Taxonomy & Distribution

| Category Name | Typical Scrap Items | Typical Weight Range | Physical Setups |
|---|---|---|---|
| **Appliance** | Industrial washing units, HVAC enclosures, compressors | $120 - 680\text{ kg}$ | 6 |
| **Automotive Scrap** | Engine blocks, door frames, axles, bonnets | $80 - 1,450\text{ kg}$ | 18 |
| **Battery** | Industrial lead-acid palletized battery packs | $150 - 350\text{ kg}$ | 4 |
| **Cardboard** | Baled high-density corrugated cardboard | $50 - 420\text{ kg}$ | 8 |
| **Cylindrical Object** | Pressurized gas cylinders, metal drums, pipe bundles | $90 - 1,245\text{ kg}$ | 7 |
| **Ferrous Metal** | Heavy steel structural scrap, iron girders, machine plates | $250 - 3,450\text{ kg}$ | 12 |
| **Grass / Biomass** | Compressed agricultural waste bales | $50 - 310\text{ kg}$ | 4 |
| **Rigid Plastic** | High-density polyethylene crates, industrial polymer tanks | $60 - 280\text{ kg}$ | 5 |
| **Rubber** | Heavy machinery tires, conveyor belt scrap | $70 - 480\text{ kg}$ | 4 |
| **Wood** | Hardwood pallets, timber beams, construction lumber | $80 - 890\text{ kg}$ | 7 |

---

## 4. Multi-View Setup & Physical Provenance

1. **Multi-View Acquisition**:
   To capture real-world visual diversity under variable perspective foreshortening and lighting, physical objects were photographed across systematically varied viewpoints (azimuths $0^\circ\text{--}360^\circ$), elevations ($D_y \in [30, 260]\text{ cm}$), and camera distances ($D_x \in [50, 450]\text{ cm}$).
2. **Fixed-Weight Multi-Angle Clusters**:
   Certain categories (such as *Industrial Gas Cylinder* and *Battery Packs*) represent multi-view captures of fixed physical setups. Each individual photo represents a distinct visual perspective, lighting condition, and camera-to-object distance.
3. **Data Quality Notes**:
   - `V_y` for Object Setup #58 contained a single typographical data-entry entry ($891\text{ cm}$ vs physical $89.1\text{ cm}$), which is handled in the preprocessing pipeline.
   - For benchmark consistency, samples with weight $< 50\text{ kg}$ (e.g. lightweight packaging debris) are excluded in `train.py` via `MIN_WEIGHT_KG = 50`.

---

## 5. Dataset Directory Format

When extracted, the dataset should follow this layout:

```
data/
├── image.csv
└── images/ (or root folder matching image_path in CSV)
    ├── Appliance/
    ├── Automotive Scrap/
    ├── Battery/
    ├── Cardboard/
    ├── Cylindrical Object/
    ├── Ferrous Metal/
    ├── Rigid Plastic/
    ├── Rubber/
    └── Wood/
```
