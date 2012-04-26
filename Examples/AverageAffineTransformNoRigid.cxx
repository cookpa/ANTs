// compute the average of a list of affine transform

#include "antsUtilities.h"
#include "itkImageFileReader.h"
#include "itkVector.h"
#include "itkImageFileWriter.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkTransformFactory.h"

#include "itkAverageAffineTransformNoRigidFunction.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

static bool AverageAffineTransformNoRigid_ParseInput(int argc, char * *argv, char *& output_transform_filename,
                                                     char *& reference_transform_filename, TRAN_OPT_QUEUE & opt_queue)
{
  opt_queue.clear();
  opt_queue.reserve(argc);

  output_transform_filename = argv[0];

  reference_transform_filename = NULL;

  int ind = 1;
  while( ind < argc )
    {
    if( strcmp(argv[ind], "-R") == 0 )
      {
      ind++;
      if( ind >= argc )
        {
        return false;
        }
      reference_transform_filename = argv[ind];
      }
    else if( strcmp(argv[ind], "-i") == 0 )
      {
      ind++;
      if( ind >= argc )
        {
        return false;
        }
      TRAN_OPT opt;
      opt.filename = argv[ind];
      if( CheckFileType(opt.filename) != AFFINE_FILE )
        {
        std::cout << "file: " << opt.filename
                  << " is not an affine .txt file. Invalid to use '-i' "
                  << std::endl;
        return false;
        }
      opt.file_type = AFFINE_FILE;
      opt.do_affine_inv = true;

      opt.weight = 1.0;       // default value
      if( ind < argc - 1 )    // test if still has extra parameters
        {
        double weight;
        if( get_a_double_number(argv[ind + 1], weight) )
          {
          ind++;
          opt.weight = weight;
          }
        }
      opt_queue.push_back(opt);
      }
    else
      {
      TRAN_OPT opt;
      opt.filename = argv[ind];
      if( CheckFileType(opt.filename) != AFFINE_FILE )
        {
        std::cout << "file: " << opt.filename
                  << " is not an affine .txt file." << std::endl;
        return false;
        }
      opt.file_type = CheckFileType(opt.filename);
      opt.do_affine_inv = false;

      opt.weight = 1.0;       // default value
      if( ind < argc - 1 )    // test if still has extra parameters
        {
        double weight;
        if( get_a_double_number(argv[ind + 1], weight) )
          {
          ind++;
          opt.weight = weight;
          }
        }

      opt_queue.push_back(opt);
      }
    ind++;
    }

//    if (reference_image_filename == NULL) {
//        std::cout << "the reference image file (-R) must be given!!!"
//        << std::endl;
//        return false;
//    }

  return true;
}

template <int ImageDimension>
void AverageAffineTransformNoRigid(char *output_affine_txt, char *reference_affine_txt,
                                   TRAN_OPT_QUEUE & opt_queue)
{
//    typedef itk::Image<float, ImageDimension> ImageType;
//    typedef itk::Vector<float, ImageDimension> VectorType;
//    typedef itk::Image<VectorType, ImageDimension> DisplacementFieldType;

  typedef itk::MatrixOffsetTransformBase<double, ImageDimension,
                                         ImageDimension> AffineTransformType;
//    typedef itk::WarpImageMultiTransformFilter<ImageType, ImageType,
//            DisplacementFieldType, AffineTransformType> WarperType;

  typedef itk::AverageAffineTransformNoRigidFunction<AffineTransformType> WarperType;

  itk::TransformFactory<AffineTransformType>::RegisterTransform();

  // typedef itk::ImageFileReader<ImageType> ImageFileReaderType;
  // typename ImageFileReaderType::Pointer reader_img = ImageFileReaderType::New();
  // typename ImageType::Pointer img_ref = ImageType::New();

  // typename ImageFileReaderType::Pointer reader_img_ref = ImageFileReaderType::New();

  WarperType average_func;
  // warper->SetInput(img_mov);
  // warper->SetEdgePaddingValue( 0);
//    VectorType pad;
//    pad.Fill(0);
  // warper->SetEdgePaddingValue(pad);

  typedef itk::TransformFileReader TranReaderType;

//    typedef itk::ImageFileReader<DisplacementFieldType> FieldReaderType;

  int       cnt_affine = 0;
  const int kOptQueueSize = opt_queue.size();
  for( int i = 0; i < kOptQueueSize; i++ )
    {
    const TRAN_OPT & opt = opt_queue[i];

    switch( opt.file_type )
      {
      case AFFINE_FILE:
        {
        typename TranReaderType::Pointer tran_reader =
          TranReaderType::New();
        tran_reader->SetFileName(opt.filename);
        tran_reader->Update();
        typename AffineTransformType::Pointer aff =
          dynamic_cast<AffineTransformType *>( (tran_reader->GetTransformList() )->front().GetPointer() );

        if(            opt_queue[i].do_affine_inv )
          {
          aff->GetInverse(aff);
          }
        // std::cout << aff << std::endl;

        double weight = opt.weight;
        average_func.PushBackAffineTransform(aff, weight);
        cnt_affine++;
        break;
        }
      case DEFORMATION_FILE:
        {
        std::cout << "Average affine only files: ignore " << opt.filename
                  << std::endl;
        break;
        }
      default:
        std::cout << "Unknown file type!" << std::endl;
      }
    }

  typedef typename WarperType::PointType PointType;
  PointType aff_center;

  typename AffineTransformType::Pointer aff_ref_tmp;
  if( reference_affine_txt )
    {
    typename TranReaderType::Pointer tran_reader = TranReaderType::New();
    tran_reader->SetFileName(reference_affine_txt);
    tran_reader->Update();
    aff_ref_tmp =
      dynamic_cast<AffineTransformType *>( (tran_reader->GetTransformList() )->front().GetPointer() );
    }
  else
    {
    if( cnt_affine > 0 )
      {
      std::cout << "the reference affine file for center is selected as the first affine!" << std::endl;
      aff_ref_tmp = average_func.GetTransformList().begin()->aff;
      }
    else
      {
      std::cout << "No affine input is given. nothing to do ......" << std::endl;
      return;
      }
    }

  aff_center = aff_ref_tmp->GetCenter();
  std::cout << "new center is : " << aff_center << std::endl;

  // warper->PrintTransformList();

  // typename AffineTransformType::Pointer aff_output = warper->ComposeAffineOnlySequence(aff_center);
  typename AffineTransformType::Pointer aff_output = AffineTransformType::New();

  average_func.AverageMultipleAffineTransform(aff_center, aff_output);

  typedef itk::TransformFileWriter TranWriterType;
  typename TranWriterType::Pointer tran_writer = TranWriterType::New();
  tran_writer->SetFileName(output_affine_txt);
  tran_writer->SetInput(aff_output);
  tran_writer->Update();

  std::cout << "wrote file to : " << output_affine_txt << std::endl;
}

int main(int argc, char * *argv)
{
  if( argc <= 3 )
    {
    std::cout
      << "AverageAffineTransform ImageDimension output_affine_transform [-R reference_affine_transform] "
      << "{[-i] affine_transform_txt [weight(=1)] ]}"
      << std::endl
      << std::endl
      << " Usage: Compute weighted average of input affine transforms. "
      << std::endl
      << "For 2D and 3D transform, the affine transform is first decomposed into "
      "scale x shearing x rotation. Then these parameters are averaged, using the weights if they provided. "
      "For 3D transform, the rotation component is the quaternion. After averaging, the quaternion will also "
      "be normalized to have unit norm. For 2D transform, the rotation component is the rotation angle. "
      "The weight for each transform is a non-negative number. The sum of all weights will be normalized to 1 "
      "before averaging. The default value for each weight is 1.0. "
      << std::endl
      << std::endl
      << "All affine transforms is a \"centerd\" transform, following ITK convention. A reference_affine_transform"
      " defines the center for the output transform. The first provided transform is the default reference "
      "transform"
      << std::endl
      << "Output affine transform is a MatrixOffsetBaseTransform."
      << std::endl
      << " -i option takes the inverse of the affine mapping."
      << std::endl
      << " For example: "
      << std::endl
      << " 2 output_affine.txt -R A.txt A1.txt 1.0 -i A2.txt 2.0 A3.txt A4.txt 6.0 A5.txt"
      << std::endl
      << "This computes: (1*A1 + 2*(A2)^-1 + A3 + A4*6 + A5 ) / (1+2+1+6+5)"
      << std::endl;
    return EXIT_SUCCESS;
    }

  TRAN_OPT_QUEUE opt_queue;

  char *output_transform_filename = NULL;
  char *reference_transform_filename = NULL;

  bool is_parsing_ok = false;
  int  kImageDim = atoi(argv[1]);

  is_parsing_ok = AverageAffineTransformNoRigid_ParseInput(argc - 2, argv + 2, output_transform_filename,
                                                           reference_transform_filename, opt_queue);

  if( is_parsing_ok )
    {
    std::cout << "output_transform_filename: " << output_transform_filename
              << std::endl;
    std::cout << "reference_transform_filename: ";

    if( reference_transform_filename )
      {
      std::cout << reference_transform_filename << std::endl;
      }
    else
      {
      std::cout << "NULL" << std::endl;
      }

    DisplayOptQueue(opt_queue);

    switch( kImageDim )
      {
      case 2:
        {
        AverageAffineTransformNoRigid<2>(output_transform_filename,
                                         reference_transform_filename, opt_queue);
        break;
        }
      case 3:
        {
        AverageAffineTransformNoRigid<3>(output_transform_filename,
                                         reference_transform_filename, opt_queue);
        break;
        }
      }
    }

  else
    {
    std::cout << "Input error!" << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}