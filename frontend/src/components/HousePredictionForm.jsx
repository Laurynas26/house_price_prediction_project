import { useState } from "react";

export default function HousePredictionForm() {
  const [formData, setFormData] = useState({
    size: "",
    contribution_vve: "",
    external_storage: "",
    nr_rooms: "",
    bathrooms: "",
    toilets: "",
    postal_code: "",
    location: "Residential",
    energy_label: "A",
    status: "Onder bod",
    roof_type: "Flat",
    ownership_type: "Owner",

    // amenities
    facilities: {
      mechanische_ventilatie: false,
      tv_kabel: false,
      lift: false,
      natuurlijke_ventilatie: false,
      schuifpui: false,
      glasvezelkabel: false,
      frans_balkon: false,
      buitenzonwering: false,
      zonnepanelen: false,
      airconditioning: false,
      domotica: false,
      sauna: false,
      zwembad: false,
    },

    balcony: false,
    backyard: false,
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleCheckbox = (group, key) => {
    setFormData((prev) => ({
      ...prev,
      [group]: {
        ...prev[group],
        [key]: !prev[group][key],
      },
    }));
  };

  const preparePayload = () => {
    const facilities = Object.entries(formData.facilities)
      .filter(([_, v]) => v)
      .map(([k]) => k);

    const outdoor_features = [];
    if (formData.balcony) outdoor_features.push("balcony");

    return {
      manual_input: {
        size: Number(formData.size),
        contribution_vve: Number(formData.contribution_vve),
        external_storage: Number(formData.external_storage),
        nr_rooms: Number(formData.nr_rooms),
        bathrooms: Number(formData.bathrooms),
        toilets: Number(formData.toilets),
        postal_code: formData.postal_code,
        location: formData.location,
        energy_label: formData.energy_label,
        status: formData.status,
        roof_type: formData.roof_type,
        ownership_type: formData.ownership_type,
        facilities,
        outdoor_features,
        backyard: formData.backyard ? "yes" : "no",
      },
    };
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = preparePayload();

    try {
      const response = await fetch(
        "https://0uytdsj3nf.execute-api.eu-north-1.amazonaws.com/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      const result = await response.json();
      if (result.success) {
        alert("Predicted price: €" + Math.round(result.prediction));
      } else {
        alert("Error: " + result.error);
      }
    } catch (err) {
      alert("Failed to fetch prediction");
      console.error(err);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <h2 className="text-xl font-bold">Basic info</h2>

      <input type="number" placeholder="Size (m²)" name="size" value={formData.size} onChange={handleChange} required />
      <input type="number" placeholder="Rooms" name="nr_rooms" value={formData.nr_rooms} onChange={handleChange} required />
      <input type="number" placeholder="Bathrooms" name="bathrooms" value={formData.bathrooms} onChange={handleChange} />
      <input type="number" placeholder="Toilets" name="toilets" value={formData.toilets} onChange={handleChange} />
      <input type="number" placeholder="VvE contribution (€)" name="contribution_vve" value={formData.contribution_vve} onChange={handleChange} />
      <input type="number" placeholder="External storage (m²)" name="external_storage" value={formData.external_storage} onChange={handleChange} />
      <input type="text" placeholder="Postal code" name="postal_code" value={formData.postal_code} onChange={handleChange} required />

      <h2 className="text-xl font-bold">Amenities</h2>
      <div className="grid grid-cols-2 gap-2">
        {Object.keys(formData.facilities).map((k) => (
          <label key={k}>
            <input type="checkbox" checked={formData.facilities[k]} onChange={() => handleCheckbox("facilities", k)} />
            {k.replace(/_/g, " ")}
          </label>
        ))}
      </div>

      <label>
        <input type="checkbox" checked={formData.balcony} onChange={() => setFormData(p => ({ ...p, balcony: !p.balcony }))} />
        Balcony
      </label>

      <label>
        <input type="checkbox" checked={formData.backyard} onChange={() => setFormData(p => ({ ...p, backyard: !p.backyard }))} />
        Backyard
      </label>

      <h2 className="text-xl font-bold">Categorical</h2>

      <select name="status" value={formData.status} onChange={handleChange}>
        <option>Onder bod</option>
        <option>Verkocht</option>
        <option>Verkocht onder voorbehoud</option>
      </select>

      <select name="roof_type" value={formData.roof_type} onChange={handleChange}>
        <option>Flat</option>
        <option>Saddle</option>
        <option>Unknown</option>
      </select>

      <select name="ownership_type" value={formData.ownership_type} onChange={handleChange}>
        <option>Owner</option>
        <option>Other</option>
        <option>Unknown</option>
      </select>

      <select name="energy_label" value={formData.energy_label} onChange={handleChange}>
        {["A", "B", "C", "D", "E", "F", "G"].map((l) => (
          <option key={l}>{l}</option>
        ))}
      </select>

      <button className="p-2 bg-blue-600 text-white">Predict price</button>
    </form>
  );
}
