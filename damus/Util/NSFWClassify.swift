//
//  NSFWClassify.swift
//  damus
//
//  Created by Jonathan on 1/29/23.
//

import UIKit
import CoreML
import Vision

public func classify(image: UIImage) -> (String, Float) {
    // Use a default model configuration.
    let defaultConfig = MLModelConfiguration()

    // Create an instance of the image classifier's wrapper class.
    let nsfwClassifierWrapper = try? nsfw(configuration: defaultConfig)

    guard let nsfwClassifier = nsfwClassifierWrapper else {
        fatalError("App failed to create an image classifier model instance.")
    }

    // Get the underlying model instance.
    let imageClassifierModel = nsfwClassifier.model

    // Create a Vision instance using the image classifier's model instance.
    guard let nsfwClassifierVisionModel = try? VNCoreMLModel(for: imageClassifierModel) else {
        fatalError("App failed to create a `VNCoreMLModel` instance.")
    }
    var top: (String, Float) = ("", 0)
    let request = VNCoreMLRequest(model: nsfwClassifierVisionModel) { (request, error) in
        if let results = request.results as? [VNClassificationObservation], let topResult = results.first {
            top = (topResult.identifier, topResult.confidence)
        }
    }
    let handler = VNImageRequestHandler(cgImage: image.cgImage!)
    try? handler.perform([request])
    return top
}
