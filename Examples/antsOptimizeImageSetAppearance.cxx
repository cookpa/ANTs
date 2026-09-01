/*=========================================================================

  Program:   Advanced Normalization Tools

  Copyright (c) ConsortiumOfANTS. All rights reserved.
  See accompanying COPYING.txt or
 https://github.com/ANTsX/ANTs/blob/main/ANTSCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "antsCommandLineParser.h"
#include "antsUtilities.h"
#include "ANTsVersion.h"

#include "itkBoxMeanImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace ants
{
namespace
{
template <typename TImage>
typename TImage::Pointer
AllocateImageLike(const TImage * reference, typename TImage::PixelType value)
{
  typename TImage::Pointer output = TImage::New();
  output->CopyInformation(reference);
  output->SetRegions(reference->GetLargestPossibleRegion());
  output->Allocate();
  output->FillBuffer(value);
  return output;
}

template <typename TImage>
typename TImage::Pointer
ReadScalarImage(const std::string & filename)
{
  using ReaderType = itk::ImageFileReader<TImage>;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  reader->Update();
  typename TImage::Pointer image = reader->GetOutput();
  image->DisconnectPipeline();
  return image;
}

template <typename TImage>
bool
HasSameGeometry(const TImage * first, const TImage * second, double tolerance = 1e-6)
{
  if (first->GetLargestPossibleRegion() != second->GetLargestPossibleRegion())
  {
    return false;
  }

  for (unsigned int d = 0; d < TImage::ImageDimension; ++d)
  {
    if (std::abs(first->GetSpacing()[d] - second->GetSpacing()[d]) > tolerance ||
        std::abs(first->GetOrigin()[d] - second->GetOrigin()[d]) > tolerance)
    {
      return false;
    }
    for (unsigned int e = 0; e < TImage::ImageDimension; ++e)
    {
      if (std::abs(first->GetDirection()[d][e] - second->GetDirection()[d][e]) > tolerance)
      {
        return false;
      }
    }
  }
  return true;
}

template <typename TImage>
bool
NormalizeToUnitRange(TImage * image, typename TImage::PixelType & minimum, typename TImage::PixelType & maximum)
{
  using CalculatorType = itk::MinimumMaximumImageCalculator<TImage>;
  typename CalculatorType::Pointer calculator = CalculatorType::New();
  calculator->SetImage(image);
  calculator->Compute();
  minimum = calculator->GetMinimum();
  maximum = calculator->GetMaximum();

  const double range = static_cast<double>(maximum) - static_cast<double>(minimum);
  if (!std::isfinite(range) || range <= std::numeric_limits<double>::epsilon())
  {
    return false;
  }

  itk::ImageRegionIterator<TImage> iterator(image, image->GetLargestPossibleRegion());
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(static_cast<typename TImage::PixelType>((static_cast<double>(iterator.Get()) - minimum) / range));
  }
  return true;
}

template <typename TImage>
typename TImage::Pointer
BoxMean(const TImage * image, unsigned int radius)
{
  using FilterType = itk::BoxMeanImageFilter<TImage, TImage>;
  typename FilterType::Pointer filter = FilterType::New();
  typename TImage::SizeType    radiusVector;
  radiusVector.Fill(radius);
  filter->SetRadius(radiusVector);
  filter->SetInput(image);
  filter->Update();
  typename TImage::Pointer output = filter->GetOutput();
  output->DisconnectPipeline();
  return output;
}

template <typename TImage>
typename TImage::Pointer
AvailableNeighborhoodCount(const TImage * image, unsigned int radius)
{
  typename TImage::Pointer output = AllocateImageLike<TImage>(image, 0);
  const auto &             region = image->GetLargestPossibleRegion();
  const auto               regionStart = region.GetIndex();
  const auto               regionSize = region.GetSize();

  itk::ImageRegionIteratorWithIndex<TImage> outputIterator(output, region);
  for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
  {
    const auto index = outputIterator.GetIndex();
    double     count = 1.0;
    for (unsigned int d = 0; d < TImage::ImageDimension; ++d)
    {
      const auto regionEnd = regionStart[d] + static_cast<itk::IndexValueType>(regionSize[d]) - 1;
      const auto first = std::max(regionStart[d], index[d] - static_cast<itk::IndexValueType>(radius));
      const auto last = std::min(regionEnd, index[d] + static_cast<itk::IndexValueType>(radius));
      count *= static_cast<double>(last - first + 1);
    }
    outputIterator.Set(static_cast<typename TImage::PixelType>(count));
  }
  return output;
}

template <typename TImage>
typename TImage::Pointer
Multiply(const TImage * first, const TImage * second)
{
  typename TImage::Pointer              output = AllocateImageLike<TImage>(first, 0);
  itk::ImageRegionConstIterator<TImage> firstIterator(first, first->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<TImage> secondIterator(second, second->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage>      outputIterator(output, output->GetLargestPossibleRegion());
  for (firstIterator.GoToBegin(), secondIterator.GoToBegin(), outputIterator.GoToBegin(); !firstIterator.IsAtEnd();
       ++firstIterator, ++secondIterator, ++outputIterator)
  {
    outputIterator.Set(firstIterator.Get() * secondIterator.Get());
  }
  return output;
}

template <typename TImage>
typename TImage::Pointer
SmoothInVoxelUnits(const TImage * image, double sigma)
{
  if (sigma <= 0.0)
  {
    typename TImage::Pointer              output = AllocateImageLike<TImage>(image, 0);
    itk::ImageRegionConstIterator<TImage> inputIterator(image, image->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TImage>      outputIterator(output, output->GetLargestPossibleRegion());
    for (inputIterator.GoToBegin(), outputIterator.GoToBegin(); !inputIterator.IsAtEnd();
         ++inputIterator, ++outputIterator)
    {
      outputIterator.Set(inputIterator.Get());
    }
    return output;
  }

  using FilterType = itk::SmoothingRecursiveGaussianImageFilter<TImage, TImage>;
  typename FilterType::Pointer        filter = FilterType::New();
  typename FilterType::SigmaArrayType sigmaArray;
  for (unsigned int d = 0; d < TImage::ImageDimension; ++d)
  {
    sigmaArray[d] = sigma * image->GetSpacing()[d];
  }
  filter->SetSigmaArray(sigmaArray);
  filter->SetInput(image);
  filter->Update();
  typename TImage::Pointer output = filter->GetOutput();
  output->DisconnectPipeline();
  return output;
}

template <typename TImage>
typename TImage::Pointer
AddScaledUpdate(const TImage * image, const TImage * update, const TImage * mask, double step)
{
  typename TImage::Pointer              output = AllocateImageLike<TImage>(image, 0);
  itk::ImageRegionConstIterator<TImage> imageIterator(image, image->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<TImage> updateIterator(update, update->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<TImage> maskIterator(mask, mask->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImage>      outputIterator(output, output->GetLargestPossibleRegion());
  for (imageIterator.GoToBegin(), updateIterator.GoToBegin(), maskIterator.GoToBegin(), outputIterator.GoToBegin();
       !imageIterator.IsAtEnd();
       ++imageIterator, ++updateIterator, ++maskIterator, ++outputIterator)
  {
    outputIterator.Set(imageIterator.Get() + step * maskIterator.Get() * updateIterator.Get());
  }
  return output;
}

template <typename TImage>
typename TImage::Pointer
CreateOtsuMask(const TImage * image, typename TImage::PixelType & threshold)
{
  using FilterType = itk::OtsuThresholdImageFilter<TImage, TImage>;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(image);
  filter->SetInsideValue(0);
  filter->SetOutsideValue(1);
  filter->Update();
  threshold = filter->GetThreshold();
  typename TImage::Pointer mask = filter->GetOutput();
  mask->DisconnectPipeline();
  return mask;
}

template <typename TImage>
struct ObjectiveAndGradient
{
  double                   Objective{ 0.0 };
  itk::SizeValueType       Count{ 0 };
  typename TImage::Pointer Gradient;
};

template <typename TImage>
ObjectiveAndGradient<TImage>
ComputeObjectiveAndGradient(const TImage *                                appearance,
                            const std::vector<typename TImage::Pointer> & images,
                            const TImage *                                mask,
                            unsigned int                                  radius,
                            double                                        varianceThreshold,
                            bool                                          computeGradient)
{
  ObjectiveAndGradient<TImage> result;
  if (computeGradient)
  {
    result.Gradient = AllocateImageLike<TImage>(appearance, 0);
  }

  double fullNeighborhoodSize = 1.0;
  for (unsigned int d = 0; d < TImage::ImageDimension; ++d)
  {
    fullNeighborhoodSize *= static_cast<double>(2 * radius + 1);
  }

  typename TImage::Pointer maskMean = BoxMean<TImage>(mask, radius);
  typename TImage::Pointer availableCount = AvailableNeighborhoodCount<TImage>(appearance, radius);
  typename TImage::Pointer maskedAppearance = Multiply<TImage>(appearance, mask);
  typename TImage::Pointer appearanceMeanNumerator = BoxMean<TImage>(maskedAppearance, radius);
  typename TImage::Pointer appearanceSquared = Multiply<TImage>(appearance, appearance);
  typename TImage::Pointer maskedAppearanceSquared = Multiply<TImage>(appearanceSquared, mask);
  typename TImage::Pointer appearanceSquaredMeanNumerator = BoxMean<TImage>(maskedAppearanceSquared, radius);

  for (const auto & image : images)
  {
    typename TImage::Pointer maskedImage = Multiply<TImage>(image, mask);
    typename TImage::Pointer imageMeanNumerator = BoxMean<TImage>(maskedImage, radius);
    typename TImage::Pointer imageSquared = Multiply<TImage>(image, image);
    typename TImage::Pointer maskedImageSquared = Multiply<TImage>(imageSquared, mask);
    typename TImage::Pointer imageSquaredMeanNumerator = BoxMean<TImage>(maskedImageSquared, radius);
    typename TImage::Pointer product = Multiply<TImage>(appearance, image);
    typename TImage::Pointer maskedProduct = Multiply<TImage>(product, mask);
    typename TImage::Pointer productMeanNumerator = BoxMean<TImage>(maskedProduct, radius);

    typename TImage::Pointer p = AllocateImageLike<TImage>(appearance, 0);
    typename TImage::Pointer q = AllocateImageLike<TImage>(appearance, 0);
    typename TImage::Pointer pTimesImageMean = AllocateImageLike<TImage>(appearance, 0);
    typename TImage::Pointer qTimesAppearanceMean = AllocateImageLike<TImage>(appearance, 0);

    itk::ImageRegionConstIterator<TImage> maskIterator(mask, mask->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> maskMeanIterator(maskMean, maskMean->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> availableCountIterator(availableCount,
                                                                 availableCount->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> appearanceMeanNumeratorIterator(
      appearanceMeanNumerator, appearanceMeanNumerator->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> appearanceSquaredMeanNumeratorIterator(
      appearanceSquaredMeanNumerator, appearanceSquaredMeanNumerator->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> imageMeanNumeratorIterator(imageMeanNumerator,
                                                                     imageMeanNumerator->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> imageSquaredMeanNumeratorIterator(
      imageSquaredMeanNumerator, imageSquaredMeanNumerator->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TImage> productMeanNumeratorIterator(
      productMeanNumerator, productMeanNumerator->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TImage> pIterator(p, p->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TImage> qIterator(q, q->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TImage> pTimesImageMeanIterator(pTimesImageMean,
                                                             pTimesImageMean->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TImage> qTimesAppearanceMeanIterator(qTimesAppearanceMean,
                                                                  qTimesAppearanceMean->GetLargestPossibleRegion());
    pIterator.GoToBegin();
    qIterator.GoToBegin();
    pTimesImageMeanIterator.GoToBegin();
    qTimesAppearanceMeanIterator.GoToBegin();

    for (maskIterator.GoToBegin(),
         maskMeanIterator.GoToBegin(),
         availableCountIterator.GoToBegin(),
         appearanceMeanNumeratorIterator.GoToBegin(),
         appearanceSquaredMeanNumeratorIterator.GoToBegin(),
         imageMeanNumeratorIterator.GoToBegin(),
         imageSquaredMeanNumeratorIterator.GoToBegin(),
         productMeanNumeratorIterator.GoToBegin();
         !maskIterator.IsAtEnd();
         ++maskIterator,
         ++maskMeanIterator,
         ++availableCountIterator,
         ++appearanceMeanNumeratorIterator,
         ++appearanceSquaredMeanNumeratorIterator,
         ++imageMeanNumeratorIterator,
         ++imageSquaredMeanNumeratorIterator,
         ++productMeanNumeratorIterator)
    {
      const double maskFraction = maskMeanIterator.Get();
      const double count = availableCountIterator.Get() * maskFraction;
      if (maskIterator.Get() > 0.5 && count > 1.0 && maskFraction > 0.0)
      {
        const double appearanceMean = appearanceMeanNumeratorIterator.Get() / maskFraction;
        const double imageMean = imageMeanNumeratorIterator.Get() / maskFraction;
        const double appearanceVariance =
          appearanceSquaredMeanNumeratorIterator.Get() / maskFraction - appearanceMean * appearanceMean;
        const double imageVariance = imageSquaredMeanNumeratorIterator.Get() / maskFraction - imageMean * imageMean;
        const double covariance = productMeanNumeratorIterator.Get() / maskFraction - appearanceMean * imageMean;

        // Count depends only on the fixed input image and mask, so objective values remain comparable
        // during line search even if the candidate appearance becomes locally constant.
        if (imageVariance > varianceThreshold)
        {
          ++result.Count;
          if (appearanceVariance > varianceThreshold)
          {
            result.Objective += covariance * covariance / (appearanceVariance * imageVariance);
            if (computeGradient)
            {
              // These are the two center-dependent coefficients in the derivative of local squared CC.
              // Dividing by the number of valid samples converts the local sums to a per-voxel derivative.
              const double pValue = 2.0 * covariance / (count * appearanceVariance * imageVariance);
              const double qValue = pValue * covariance / appearanceVariance;
              pIterator.Set(pValue);
              qIterator.Set(qValue);
              pTimesImageMeanIterator.Set(pValue * imageMean);
              qTimesAppearanceMeanIterator.Set(qValue * appearanceMean);
            }
          }
        }
      }
      ++pIterator;
      ++qIterator;
      ++pTimesImageMeanIterator;
      ++qTimesAppearanceMeanIterator;
    }

    if (computeGradient)
    {
      // A voxel contributes to every CC neighborhood that contains it.  Applying the symmetric box
      // operator a second time accumulates those overlapping contributions (the adjoint operation).
      // BoxMean divides by the cropped neighborhood size at image boundaries, so the loop below
      // restores that spatially varying factor before applying the constant full-window normalization.
      typename TImage::Pointer pMean = BoxMean<TImage>(p, radius);
      typename TImage::Pointer qMean = BoxMean<TImage>(q, radius);
      typename TImage::Pointer pTimesImageMeanMean = BoxMean<TImage>(pTimesImageMean, radius);
      typename TImage::Pointer qTimesAppearanceMeanMean = BoxMean<TImage>(qTimesAppearanceMean, radius);

      itk::ImageRegionConstIterator<TImage> appearanceIterator(appearance, appearance->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> imageIterator(image, image->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> pMeanIterator(pMean, pMean->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> qMeanIterator(qMean, qMean->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> pTimesImageMeanMeanIterator(
        pTimesImageMeanMean, pTimesImageMeanMean->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> qTimesAppearanceMeanMeanIterator(
        qTimesAppearanceMeanMean, qTimesAppearanceMeanMean->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> gradientMaskIterator(mask, mask->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<TImage> gradientAvailableCountIterator(availableCount,
                                                                           availableCount->GetLargestPossibleRegion());
      itk::ImageRegionIterator<TImage> gradientIterator(result.Gradient, result.Gradient->GetLargestPossibleRegion());
      for (appearanceIterator.GoToBegin(),
           imageIterator.GoToBegin(),
           pMeanIterator.GoToBegin(),
           qMeanIterator.GoToBegin(),
           pTimesImageMeanMeanIterator.GoToBegin(),
           qTimesAppearanceMeanMeanIterator.GoToBegin(),
           gradientMaskIterator.GoToBegin(),
           gradientAvailableCountIterator.GoToBegin(),
           gradientIterator.GoToBegin();
           !appearanceIterator.IsAtEnd();
           ++appearanceIterator,
           ++imageIterator,
           ++pMeanIterator,
           ++qMeanIterator,
           ++pTimesImageMeanMeanIterator,
           ++qTimesAppearanceMeanMeanIterator,
           ++gradientMaskIterator,
           ++gradientAvailableCountIterator,
           ++gradientIterator)
      {
        if (gradientMaskIterator.Get() > 0.5)
        {
          const double adjointScale = gradientAvailableCountIterator.Get() / fullNeighborhoodSize;
          const double derivative =
            adjointScale * (imageIterator.Get() * pMeanIterator.Get() - pTimesImageMeanMeanIterator.Get() -
                            appearanceIterator.Get() * qMeanIterator.Get() + qTimesAppearanceMeanMeanIterator.Get());
          gradientIterator.Set(gradientIterator.Get() + derivative / images.size());
        }
      }
    }
  }

  if (result.Count > 0)
  {
    result.Objective /= static_cast<double>(result.Count);
  }
  return result;
}

template <unsigned int Dimension>
int
OptimizeAppearance(itk::ants::CommandLineParser * parser)
{
  using PixelType = double;
  using ImageType = itk::Image<PixelType, Dimension>;

  const auto inputOption = parser->GetOption("input");
  const auto outputOption = parser->GetOption("output");
  if (!inputOption || inputOption->GetNumberOfFunctions() < 2)
  {
    std::cerr << "At least two input images must be specified with repeated -i options." << std::endl;
    return EXIT_FAILURE;
  }
  if (!outputOption || outputOption->GetNumberOfFunctions() == 0)
  {
    std::cerr << "An output image must be specified with -o." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> inputNames;
  inputNames.reserve(inputOption->GetNumberOfFunctions());
  for (unsigned int i = 0; i < inputOption->GetNumberOfFunctions(); ++i)
  {
    inputNames.push_back(inputOption->GetFunction(i)->GetName());
  }
  const std::string outputName = outputOption->GetFunction(0)->GetName();

  unsigned int numberOfIterations = 5;
  double       convergenceThreshold = 0.0;
  const auto   convergenceOption = parser->GetOption("convergence");
  if (convergenceOption && convergenceOption->GetNumberOfFunctions())
  {
    if (convergenceOption->GetFunction(0)->GetNumberOfParameters() == 0)
    {
      numberOfIterations = parser->Convert<unsigned int>(convergenceOption->GetFunction(0)->GetName());
    }
    else
    {
      numberOfIterations = parser->Convert<unsigned int>(convergenceOption->GetFunction(0)->GetParameter(0));
      if (convergenceOption->GetFunction(0)->GetNumberOfParameters() > 1)
      {
        convergenceThreshold = parser->Convert<double>(convergenceOption->GetFunction(0)->GetParameter(1));
      }
    }
  }

  unsigned int radius = 4;
  const auto   radiusOption = parser->GetOption("radius");
  if (radiusOption && radiusOption->GetNumberOfFunctions())
  {
    radius = parser->Convert<unsigned int>(radiusOption->GetFunction(0)->GetName());
  }

  double     gradientStep = 0.1;
  const auto gradientOption = parser->GetOption("gradient-step");
  if (gradientOption && gradientOption->GetNumberOfFunctions())
  {
    gradientStep = parser->Convert<double>(gradientOption->GetFunction(0)->GetName());
  }

  double     smoothingSigma = 1.0;
  const auto smoothingOption = parser->GetOption("smoothing-sigma");
  if (smoothingOption && smoothingOption->GetNumberOfFunctions())
  {
    smoothingSigma = parser->Convert<double>(smoothingOption->GetFunction(0)->GetName());
  }

  double     varianceThreshold = 1e-6;
  const auto varianceOption = parser->GetOption("variance-threshold");
  if (varianceOption && varianceOption->GetNumberOfFunctions())
  {
    varianceThreshold = parser->Convert<double>(varianceOption->GetFunction(0)->GetName());
  }

  bool       verbose = false;
  const auto verboseOption = parser->GetOption("verbose");
  if (verboseOption && verboseOption->GetNumberOfFunctions())
  {
    verbose = parser->Convert<bool>(verboseOption->GetFunction(0)->GetName());
  }

  if (numberOfIterations == 0 || gradientStep <= 0.0 || smoothingSigma < 0.0 || varianceThreshold < 0.0)
  {
    std::cerr << "Iterations and gradient step must be positive; smoothing sigma and variance threshold must be "
                 "non-negative."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<typename ImageType::Pointer> images;
  images.reserve(inputNames.size());
  typename ImageType::Pointer appearance;
  for (unsigned int imageIndex = 0; imageIndex < inputNames.size(); ++imageIndex)
  {
    typename ImageType::Pointer image = ReadScalarImage<ImageType>(inputNames[imageIndex]);
    if (imageIndex == 0)
    {
      appearance = AllocateImageLike<ImageType>(image, 0);
    }
    else if (!HasSameGeometry<ImageType>(appearance, image))
    {
      std::cerr << "Input image geometry does not match the first image: " << inputNames[imageIndex] << std::endl;
      return EXIT_FAILURE;
    }

    PixelType minimum;
    PixelType maximum;
    if (!NormalizeToUnitRange<ImageType>(image, minimum, maximum))
    {
      std::cerr << "Input image has a non-finite or constant intensity range: " << inputNames[imageIndex] << std::endl;
      return EXIT_FAILURE;
    }

    itk::ImageRegionConstIterator<ImageType> inputIterator(image, image->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType>      appearanceIterator(appearance, appearance->GetLargestPossibleRegion());
    for (inputIterator.GoToBegin(), appearanceIterator.GoToBegin(); !inputIterator.IsAtEnd();
         ++inputIterator, ++appearanceIterator)
    {
      appearanceIterator.Set(appearanceIterator.Get() + inputIterator.Get() / inputNames.size());
    }
    images.push_back(image);
  }

  bool       useOtsuMask = false;
  const auto maskOption = parser->GetOption("mask");
  if (maskOption && maskOption->GetNumberOfFunctions())
  {
    std::string maskMode = maskOption->GetFunction(0)->GetName();
    std::transform(maskMode.begin(), maskMode.end(), maskMode.begin(), [](unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
    if (maskMode == "otsu" || maskMode == "1")
    {
      useOtsuMask = true;
    }
    else if (maskMode != "none" && maskMode != "0")
    {
      std::cerr << "Unsupported mask mode '" << maskMode << "'; use Otsu or none." << std::endl;
      return EXIT_FAILURE;
    }
  }

  typename ImageType::Pointer mask;
  if (useOtsuMask)
  {
    PixelType threshold;
    mask = CreateOtsuMask<ImageType>(appearance, threshold);
    if (verbose)
    {
      std::cout << "Otsu foreground mask threshold = " << threshold << std::endl;
    }
  }
  else
  {
    mask = AllocateImageLike<ImageType>(appearance, 1);
  }

  auto current = ComputeObjectiveAndGradient<ImageType>(appearance, images, mask, radius, varianceThreshold, true);
  if (current.Count == 0)
  {
    std::cerr << "No masked neighborhoods had sufficient intensity variance to compute an appearance update."
              << std::endl;
    return EXIT_FAILURE;
  }
  if (verbose)
  {
    std::cout << "Iteration 0: mean local squared correlation = " << current.Objective << std::endl;
  }

  constexpr unsigned int maximumBacktrackingSteps = 12;
  for (unsigned int iteration = 0; iteration < numberOfIterations; ++iteration)
  {
    typename ImageType::Pointer     smoothedGradient = SmoothInVoxelUnits<ImageType>(current.Gradient, smoothingSigma);
    double                          acceptedStep = gradientStep;
    typename ImageType::Pointer     acceptedAppearance;
    ObjectiveAndGradient<ImageType> accepted;
    bool                            updateAccepted = false;

    for (unsigned int backtracking = 0; backtracking < maximumBacktrackingSteps; ++backtracking)
    {
      typename ImageType::Pointer candidate =
        AddScaledUpdate<ImageType>(appearance, smoothedGradient, mask, acceptedStep);
      auto candidateResult =
        ComputeObjectiveAndGradient<ImageType>(candidate, images, mask, radius, varianceThreshold, false);
      if (candidateResult.Count > 0 && candidateResult.Objective > current.Objective)
      {
        acceptedAppearance = candidate;
        accepted = candidateResult;
        updateAccepted = true;
        break;
      }
      acceptedStep *= 0.5;
    }

    if (!updateAccepted)
    {
      if (verbose)
      {
        std::cout << "Stopped: no objective-improving step was found after " << maximumBacktrackingSteps
                  << " backtracking attempts." << std::endl;
      }
      break;
    }

    const double objectiveChange = accepted.Objective - current.Objective;
    appearance = acceptedAppearance;
    if (verbose)
    {
      std::cout << "Iteration " << iteration + 1 << ": mean local squared correlation = " << accepted.Objective
                << ", step = " << acceptedStep << std::endl;
    }
    if (convergenceThreshold > 0.0 && objectiveChange <= convergenceThreshold)
    {
      if (verbose)
      {
        std::cout << "Converged: objective improvement did not exceed " << convergenceThreshold << std::endl;
      }
      break;
    }
    current = ComputeObjectiveAndGradient<ImageType>(appearance, images, mask, radius, varianceThreshold, true);
  }

  using WriterType = itk::ImageFileWriter<ImageType>;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(outputName);
  writer->SetInput(appearance);
  writer->Update();
  return EXIT_SUCCESS;
}

void
InitializeCommandLineOptions(itk::ants::CommandLineParser * parser)
{
  using OptionType = itk::ants::CommandLineParser::OptionType;

  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("dimensionality");
    option->SetShortName('d');
    option->SetUsageOption(0, "2/3");
    option->SetDescription("Image dimensionality. Default = 3.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("input");
    option->SetShortName('i');
    option->SetUsageOption(0, "warpedImage");
    option->SetDescription("Input image already resampled into template space. Repeat -i for every image.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("output");
    option->SetShortName('o');
    option->SetUsageOption(0, "outputTemplate");
    option->SetDescription("Output optimized appearance image.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("convergence");
    option->SetShortName('c');
    option->SetUsageOption(0, "numberOfIterations");
    option->SetUsageOption(1, "[numberOfIterations=5,convergenceThreshold=0]");
    option->SetDescription("Maximum iterations and optional absolute objective-change threshold.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("radius");
    option->SetShortName('r');
    option->SetUsageOption(0, "4");
    option->SetDescription("Isotropic local cross-correlation neighborhood radius in voxels. Default = 4.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("gradient-step");
    option->SetShortName('g');
    option->SetUsageOption(0, "0.1");
    option->SetDescription("Appearance-gradient ascent step. Default = 0.1.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("smoothing-sigma");
    option->SetShortName('s');
    option->SetUsageOption(0, "1.0");
    option->SetDescription("Gaussian sigma applied to the mean appearance update, in voxels. Default = 1.0.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("mask");
    option->SetShortName('x');
    option->SetUsageOption(0, "Otsu");
    option->SetUsageOption(1, "none");
    option->SetDescription(
      "Optional automatic foreground mask computed from the initial normalized mean. Default = none.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("variance-threshold");
    option->SetUsageOption(0, "1e-6");
    option->SetDescription("Neighborhoods with variance at or below this value are skipped. Default = 1e-6.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("verbose");
    option->SetShortName('v');
    option->SetUsageOption(0, "0/1");
    option->SetDescription("Print the appearance objective at each iteration.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("version");
    option->SetDescription("Print version information.");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetShortName('h');
    option->SetDescription("Print the short help menu.");
    option->AddFunction("0");
    parser->AddOption(option);
  }
  {
    OptionType::Pointer option = OptionType::New();
    option->SetLongName("help");
    option->SetDescription("Print the help menu.");
    option->AddFunction("0");
    parser->AddOption(option);
  }
}
} // namespace

int
antsOptimizeImageSetAppearance(std::vector<std::string> args, std::ostream *)
{
  args.insert(args.begin(), "antsOptimizeImageSetAppearance");
  const int           argc = static_cast<int>(args.size());
  std::vector<char *> argv(args.size() + 1, nullptr);
  for (std::size_t i = 0; i < args.size(); ++i)
  {
    argv[i] = &args[i][0];
  }

  itk::ants::CommandLineParser::Pointer parser = itk::ants::CommandLineParser::New();
  parser->SetCommand(argv[0]);
  parser->SetCommandDescription(
    "Optimize the appearance of an aligned image set by gradient ascent on local squared cross-correlation, as "
    "described for SyGN template construction. Inputs are expected to have already been resampled into template "
    "space.");
  InitializeCommandLineOptions(parser);

  if (parser->Parse(argc, argv.data()) == EXIT_FAILURE)
  {
    return EXIT_FAILURE;
  }
  if (argc == 1)
  {
    parser->PrintMenu(std::cerr, 5, false);
    return EXIT_FAILURE;
  }
  if (parser->GetOption("help")->GetFunction() &&
      parser->Convert<bool>(parser->GetOption("help")->GetFunction()->GetName()))
  {
    parser->PrintMenu(std::cout, 5, false);
    return EXIT_SUCCESS;
  }
  if (parser->GetOption('h')->GetFunction() && parser->Convert<bool>(parser->GetOption('h')->GetFunction()->GetName()))
  {
    parser->PrintMenu(std::cout, 5, true);
    return EXIT_SUCCESS;
  }
  const auto versionOption = parser->GetOption("version");
  if (versionOption && versionOption->GetNumberOfFunctions())
  {
    std::cout << ANTs::Version::ExtendedVersionString() << std::endl;
    return EXIT_SUCCESS;
  }

  unsigned int dimension = 3;
  const auto   dimOption = parser->GetOption("dimensionality");
  if (dimOption && dimOption->GetNumberOfFunctions())
  {
    dimension = parser->Convert<unsigned int>(dimOption->GetFunction(0)->GetName());
  }

  try
  {
    switch (dimension)
    {
      case 2:
        return OptimizeAppearance<2>(parser);
      case 3:
        return OptimizeAppearance<3>(parser);
      default:
        std::cerr << "Unsupported dimensionality " << dimension << "; use 2 or 3." << std::endl;
        return EXIT_FAILURE;
    }
  }
  catch (const itk::ExceptionObject & error)
  {
    std::cerr << "antsOptimizeImageSetAppearance failed: " << error << std::endl;
    return EXIT_FAILURE;
  }
  catch (const std::exception & error)
  {
    std::cerr << "antsOptimizeImageSetAppearance failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
} // namespace ants
