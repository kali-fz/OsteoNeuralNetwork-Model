/**
 * ISO 3166-1 alpha-2 -> the country's feature id in countries-110m.json.
 *
 * GENERATED. Regenerate if either the centroid table or the map file changes;
 * tests/globe-countries.test.mjs fails if the two drift apart.
 *
 * WHY THIS EXISTS
 * ---------------
 * The globe used to decide which shape to fill by asking which polygon
 * contained a marker's coordinate (d3.geoContains). That is geometry answering
 * a question about identity, and it got the answer wrong:
 *
 *   - a marker nudged even slightly outside its own border filled the
 *     neighbour instead -- Belgium's marker filled the whole of France,
 *     including French Guiana on the other side of the Atlantic;
 *   - Singapore's centroid sits inside Malaysia's polygon at this resolution,
 *     so Singapore filled Malaysia;
 *   - Bahrain, Malta, Mauritius, the Maldives, New Zealand and the Philippines
 *     have centroids that fall in water between islands, so they filled nothing.
 *
 * A country code is already carried on every marker, so the shape is looked up
 * by identity instead. Geometry is now only used to draw.
 *
 * The countries below have no polygon in the 110m map at all, so they are
 * absent here by necessity rather than by oversight. Their dot and their
 * tooltip still appear; only the filled shape is unavailable:
 *
 *   BH Bahrain
 *   HK Hong Kong
 *   MT Malta
 *   MU Mauritius
 *   MV Maldives
 *   SG Singapore
 */
export const COUNTRY_MAP_IDS = {
  "AE": "784", // United Arab Emirates
  "AF": "004", // Afghanistan
  "AL": "008", // Albania
  "AM": "051", // Armenia
  "AO": "024", // Angola
  "AR": "032", // Argentina
  "AT": "040", // Austria
  "AU": "036", // Australia
  "AZ": "031", // Azerbaijan
  "BA": "070", // Bosnia and Herzegovina
  "BD": "050", // Bangladesh
  "BE": "056", // Belgium
  "BF": "854", // Burkina Faso
  "BG": "100", // Bulgaria
  "BI": "108", // Burundi
  "BJ": "204", // Benin
  "BO": "068", // Bolivia
  "BR": "076", // Brazil
  "BW": "072", // Botswana
  "BY": "112", // Belarus
  "CA": "124", // Canada
  "CD": "180", // DR Congo
  "CF": "140", // Central African Republic
  "CG": "178", // Congo
  "CH": "756", // Switzerland
  "CI": "384", // Cote d'Ivoire
  "CL": "152", // Chile
  "CM": "120", // Cameroon
  "CN": "156", // China
  "CO": "170", // Colombia
  "CR": "188", // Costa Rica
  "CU": "192", // Cuba
  "CY": "196", // Cyprus
  "CZ": "203", // Czechia
  "DE": "276", // Germany
  "DK": "208", // Denmark
  "DO": "214", // Dominican Republic
  "DZ": "012", // Algeria
  "EC": "218", // Ecuador
  "EE": "233", // Estonia
  "EG": "818", // Egypt
  "ES": "724", // Spain
  "ET": "231", // Ethiopia
  "FI": "246", // Finland
  "FR": "250", // France
  "GA": "266", // Gabon
  "GB": "826", // United Kingdom
  "GE": "268", // Georgia
  "GH": "288", // Ghana
  "GN": "324", // Guinea
  "GR": "300", // Greece
  "GT": "320", // Guatemala
  "HN": "340", // Honduras
  "HR": "191", // Croatia
  "HU": "348", // Hungary
  "ID": "360", // Indonesia
  "IE": "372", // Ireland
  "IL": "376", // Israel
  "IN": "356", // India
  "IQ": "368", // Iraq
  "IR": "364", // Iran
  "IS": "352", // Iceland
  "IT": "380", // Italy
  "JM": "388", // Jamaica
  "JO": "400", // Jordan
  "JP": "392", // Japan
  "KE": "404", // Kenya
  "KG": "417", // Kyrgyzstan
  "KH": "116", // Cambodia
  "KR": "410", // South Korea
  "KW": "414", // Kuwait
  "KZ": "398", // Kazakhstan
  "LA": "418", // Laos
  "LB": "422", // Lebanon
  "LK": "144", // Sri Lanka
  "LT": "440", // Lithuania
  "LU": "442", // Luxembourg
  "LV": "428", // Latvia
  "LY": "434", // Libya
  "MA": "504", // Morocco
  "MD": "498", // Moldova
  "ME": "499", // Montenegro
  "MG": "450", // Madagascar
  "MK": "807", // North Macedonia
  "ML": "466", // Mali
  "MM": "104", // Myanmar
  "MN": "496", // Mongolia
  "MW": "454", // Malawi
  "MX": "484", // Mexico
  "MY": "458", // Malaysia
  "MZ": "508", // Mozambique
  "NA": "516", // Namibia
  "NE": "562", // Niger
  "NG": "566", // Nigeria
  "NI": "558", // Nicaragua
  "NL": "528", // Netherlands
  "NO": "578", // Norway
  "NP": "524", // Nepal
  "NZ": "554", // New Zealand
  "OM": "512", // Oman
  "PA": "591", // Panama
  "PE": "604", // Peru
  "PH": "608", // Philippines
  "PK": "586", // Pakistan
  "PL": "616", // Poland
  "PS": "275", // Palestine
  "PT": "620", // Portugal
  "PY": "600", // Paraguay
  "QA": "634", // Qatar
  "RO": "642", // Romania
  "RS": "688", // Serbia
  "RU": "643", // Russia
  "RW": "646", // Rwanda
  "SA": "682", // Saudi Arabia
  "SD": "729", // Sudan
  "SE": "752", // Sweden
  "SI": "705", // Slovenia
  "SK": "703", // Slovakia
  "SN": "686", // Senegal
  "SO": "706", // Somalia
  "SV": "222", // El Salvador
  "SY": "760", // Syria
  "TG": "768", // Togo
  "TH": "764", // Thailand
  "TN": "788", // Tunisia
  "TR": "792", // Turkiye
  "TT": "780", // Trinidad and Tobago
  "TW": "158", // Taiwan
  "TZ": "834", // Tanzania
  "UA": "804", // Ukraine
  "UG": "800", // Uganda
  "US": "840", // United States
  "UY": "858", // Uruguay
  "UZ": "860", // Uzbekistan
  "VE": "862", // Venezuela
  "VN": "704", // Vietnam
  "YE": "887", // Yemen
  "ZA": "710", // South Africa
  "ZM": "894", // Zambia
  "ZW": "716", // Zimbabwe
};
