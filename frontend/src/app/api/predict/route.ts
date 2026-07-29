import { NextResponse } from 'next/server';
import modelData from './model.json';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Extract KNN model data
    const featureCols = modelData.feature_cols;
    const catFeatures = modelData.cat_features;
    const numFeatures = modelData.num_features;
    
    const medians = modelData.medians as Record<string, number>;
    const modes = modelData.modes as Record<string, string>;
    const categoryMappings = modelData.category_mappings as Record<string, Record<string, number>>;
    const means = modelData.means;
    const stds = modelData.stds;
    
    const X_train = modelData.X_train;
    const y_train = modelData.y_train;

    // 1. Impute missing numerical and categorical values
    const imputedData: Record<string, number | string> = {};
    for (const col of numFeatures) {
      imputedData[col] = body[col] !== undefined && body[col] !== null ? Number(body[col]) : medians[col];
    }
    for (const col of catFeatures) {
      imputedData[col] = body[col] !== undefined && body[col] !== null ? String(body[col]) : modes[col];
    }

    // 2. Encode categorical features
    const encodedData: Record<string, number> = {};
    for (const col of catFeatures) {
      const mapping = categoryMappings[col];
      let val = String(imputedData[col]);
      if (mapping[val] === undefined) {
        val = modes[col];
      }
      encodedData[col] = mapping[val];
    }

    // Copy numerical values
    for (const col of numFeatures) {
      encodedData[col] = Number(imputedData[col]);
    }

    // 3. Align features in the exact column order expected by the model
    const x_raw: number[] = [];
    for (const col of featureCols) {
      x_raw.push(encodedData[col]);
    }

    // 4. Standard scale features using mean and standard deviation
    const x_scaled = x_raw.map((val, idx) => (val - means[idx]) / stds[idx]);

    // 5. Run KNN classification (K = 13)
    // Compute Euclidean distances
    const distances = X_train.map((trainRow) => {
      let sumSq = 0;
      for (let j = 0; j < x_scaled.length; j++) {
        sumSq += Math.pow(trainRow[j] - x_scaled[j], 2);
      }
      return Math.sqrt(sumSq);
    });

    // Pair distances with their labels
    const distancePairs = distances.map((dist, idx) => ({ dist, label: y_train[idx] }));
    
    // Sort distances in ascending order
    distancePairs.sort((a, b) => a.dist - b.dist);

    // Get the top 13 nearest neighbors
    const k = 13;
    const nearestNeighbors = distancePairs.slice(0, k);

    // Calculate approval probability
    const approvals = nearestNeighbors.filter(item => item.label === 1).length;
    const approvalProb = approvals / k;

    // Final decision
    const prediction = approvalProb >= 0.5 ? "Approved" : "Rejected";
    const confidence = prediction === "Approved" ? approvalProb : (1.0 - approvalProb);

    return NextResponse.json({
      prediction,
      probability: approvalProb,
      confidence,
      features_received: body
    });
  } catch (error: any) {
    console.error("API error:", error);
    return NextResponse.json({ detail: `Prediction error: ${error.message}` }, { status: 500 });
  }
}
