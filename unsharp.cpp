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
    IppiBorderSize sobBorderSize = iwiSizeToBorderSize(iwiMaskToSize(ippMskSize3x3)); // Convert mask size to border size

    #pragma omp parallel num_threads(threads)
    {
        // Declare thread-scope variables
        IppiBorderType border;
        ipp::IwiImage srcTile, cvtTile, dstTile;
        Ipp8u *pTileBuffer = NULL;

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
            ipp::iwiRoi_CorrectBordersOverlap(sobBorderSize, cvtImage.m_size, &tile); // Check borders overlap and correct tile of necessary
            border = ipp::iwiRoi_GetTileBorder(ippBorderRepl, sobBorderSize, cvtImage.m_size, tile); // Get actual tile border

            // Get images for current ROI
            cvtTile = cvtImage.GetRoiImage(tile);
            dstTile = dstImage.GetRoiImage(tile);

            // Run functions
            //ipp::iwiFilterSobel(&cvtTile, &dstTile, iwiDerivHorFirst, ippMskSize3x3, border);
            ippiFilterSharpenBorderGetBufferSize(tileSize, ippMskSize3x3, ipp8u, ipp16s, 1, pTileBuffer);
            ippiFilterSharpenBorder_8u_C3R(&cvtTile, 0, &dstTile, 0, ippMskSize3x3, border, pTileBuffer);
        }
    }
}
