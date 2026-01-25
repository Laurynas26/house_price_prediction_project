import { useState } from "react";

const FACILITY_LABELS = {
  mechanische_ventilatie: "Mechanical ventilation",
  natuurlijke_ventilatie: "Natural ventilation",
  tv_kabel: "Cable TV",
  glasvezelkabel: "Fiber internet",
  lift: "Elevator",
  schuifpui: "Sliding doors",
  frans_balkon: "French balcony",
  buitenzonwering: "Outdoor sun shades",
  zonnepanelen: "Solar panels",
  airconditioning: "Air conditioning",
  domotica: "Smart home (domotics)",
  sauna: "Sauna",
  zwembad: "Swimming pool",
};

export default function HousePredictionForm() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

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
    bedrooms: "",
    year_of_construction: "",
    located_on: "",

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
      .map(([k]) => k)
      .join(","); // safer than array if model expects string

    return {
      manual_input: {
        size: Number(formData.size) || null,
        contribution_vve: Number(formData.contribution_vve) || null,
        external_storage: Number(formData.external_storage) || null,
        nr_rooms: Number(formData.nr_rooms) || null,
        bathrooms: Number(formData.bathrooms) || null,
        toilets: Number(formData.toilets) || null,
        bedrooms: Number(formData.bedrooms) || null,
        year_of_construction: Number(formData.year_of_construction) || null,

        postal_code: formData.postal_code || null,
        location: formData.location,
        energy_label: formData.energy_label,
        status: formData.status,
        roof_type: formData.roof_type,
        ownership_type: formData.ownership_type,
        located_on: Number(formData.located_on) || null,

        facilities,

        outdoor_features: {
          balcony: formData.balcony,
          backyard: formData.backyard,
        },

        balcony: formData.balcony ? "yes" : "no",
        backyard: formData.backyard ? "yes" : "no",
      },
    };
  };


  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = preparePayload();

    setLoading(true);
    setPrediction(null);
    setError(null);

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
        setPrediction(Math.round(result.prediction));
      } else {
        setError(result.error || "Prediction failed");
      }
    } catch (err) {
      setError("Failed to fetch prediction");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <h2 className="text-xl font-bold">Basic info</h2>
      <label className="block">
        <span className="text-sm text-gray-600">Size (m²)</span>
        <input
          type="number"
          name="size"
          value={formData.size}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Rooms</span>
        <input
          type="number"
          name="nr_rooms"
          value={formData.nr_rooms}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Bathrooms</span>
        <input
          type="number"
          name="bathrooms"
          value={formData.bathrooms}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Toilets</span>
        <input
          type="number"
          name="toilets"
          value={formData.toilets}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">VvE contribution (€)</span>
        <input
          type="number"
          name="contribution_vve"
          value={formData.contribution_vve}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">External storage (m²)</span>
        <input
          type="number"
          name="external_storage"
          value={formData.external_storage}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Postal code</span>
        <input
          type="text"
          name="postal_code"
          value={formData.postal_code}
          onChange={handleChange}
          required
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Year of construction</span>
        <input
          type="number"
          name="year_of_construction"
          value={formData.year_of_construction}
          onChange={handleChange}
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Bedrooms</span>
        <input
          type="number"
          name="bedrooms"
          value={formData.bedrooms}
          onChange={handleChange}
          className="w-full"
        />
      </label>
      <label className="block">
        <span className="text-sm text-gray-600">Located on (floor)</span>
        <input
          type="number"
          name="located_on"
          value={formData.located_on}
          onChange={handleChange}
          className="w-full"
        />
      </label>



      <h2 className="text-xl font-bold">Amenities</h2>
      <div className="grid grid-cols-2 gap-2">
        {Object.keys(formData.facilities).map((k) => (
          <label key={k} className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={formData.facilities[k]}
              onChange={() => handleCheckbox("facilities", k)}
            />
            {FACILITY_LABELS[k] ?? k.replace(/_/g, " ")}
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
        <option value="Flat">Flat roof</option>
        <option value="Saddle">Saddle roof</option>
        <option value="Unknown">Unknown</option>
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

      <button
        className="p-2 bg-blue-600 text-white disabled:opacity-50"
        disabled={loading}
      >
        {loading ? "Predicting..." : "Predict price"}
      </button>

      {prediction && (
        <div className="mt-4 p-4 bg-green-100 rounded">
          <p className="text-sm text-gray-600">Estimated price</p>
          <p className="text-2xl font-bold">
            € {prediction.toLocaleString()}
          </p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
    </form>
  );
}
