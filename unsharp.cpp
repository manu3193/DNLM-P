#include "iw++/iw.hpp"
#include <ipp.h>
#include <omp.h>

int main(int, char**)
{
    // Create images
    ipp::IwiImage srcImage, cvtImage, dstImage;
    srcImage.Alloc(ipp::IwiSize(320, 240), ipp8u, 3);
    cvtImage.Alloc(srcImage.m_size, ipp8u, 1);
    dstImage.Alloc(srcImage.m_size, ipp16s, 1);

    int            threads = omp_get_max_threads(); // Get threads number
    ipp::IwiSize   tileSize(dstImage.m_size.width, (dstImage.m_size.height + threads - 1)/threads); // One tile per thread
    IppiSize       kernelSize = {3, 3};
    IppiBorderSize borderSize = iwiSizeToBorderSize(iwiMaskToSize(ippMskSize3x3)); // Convert kernel size to border size
    
    const Ipp16s   kernel[3*3] = {-1/8, -1/8, -1/8, -1/8, 16/8, -1/8, -1/8, -1/8, -1/8}; // Define high pass filter
    int   numberChannels = 1; //Number of channels of the cvt and src tile


    #pragma omp parallel num_threads(threads)
    {
        // Declare thread-scope variables
        IppiBorderType        border;
        ipp::IwiImage         srcTile, cvtTile, dstTile;
        int                   filterBufferSize = 0, filterSpecSize = 0; // Size of the work buffer and filter specification structure required for filtering
        int                   srcStep = 0, dstStep = 0; //Steps in bytes through src and dst images
        IppiFilterBorderSpec  *pFilterSpec = NULL;   //Filter specification context structure 
        Ipp16s                *pFilterBuffer = NULL; //Filter pointer to buffer

        // Color convert threading
        #pragma omp for
        for(IppSizeL row = 0; row < dstImage.m_size.height; row += tileSize.height)
        {
            ipp::IwiRect tile(0, row, tileSize.width, tileSize.height); // Create actual tile rectangle

            // Get images for current ROI
            srcTile = srcImage.GetRoiImage(tile);
            cvtTile = cvtImage.GetRoiImage(tile);

            // Run functions
            ipp::iwiColorConvert(&srcTile, iwiColorRGB, &cvtTile, iwiColorGray);
        }

        // Sobel threading
        #pragma omp for
        for(IppSizeL row = 0; row < dstImage.m_size.height; row += tileSize.height)
        {
            ipp::IwiRect tile(0, row, tileSize.width, tileSize.height); // Create actual tile rectangle
            border = ipp::iwiRoi_GetTileBorder(ippBorderRepl, borderSize, cvtImage.m_size, tile); // Get actual tile border
            ipp::iwiRoi_CorrectBordersOverlap(border, borderSize, cvtImage.m_size, &tile); // Check borders overlap and correct tile of necessary

            // Get images for current ROI
            cvtTile = cvtImage.GetRoiImage(tile);
            dstTile = dstImage.GetRoiImage(tile);

            // Run functions
            //ipp::iwiFilterSobel(&cvtTile, &dstTile, iwiDerivHorFirst, ippMskSize3x3, border);
            ippiFilterBorderGetSize(kernelSize, {dstTile.m_size.width, dstTile.m_size.width} , ipp16s, ipp16s, 1, &filterSpecSize, &filterBufferSize);
            pFilterSpec = (IppiFilterBorderSpec *)ippsMalloc_16s(filterSpecSize);
            pFilterBuffer = ippsMalloc_16s(filterBufferSize);
            ippiFilterBorderInit_16s(kernel, kernelSize, 1, ipp8u, numberChannels, ippRndHintAccurate, pFilterSpec);
            ippiFilterBorder_16s_C1R(cvtTile.m_ptr, srcStep, dstTile.m_ptr, dstStep, {dstTile.m_size.width, dstTile.m_size.width}, border, 0, pFilterSpec, pFilterBuffer );
        }
    }
}
