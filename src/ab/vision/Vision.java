/*****************************************************************************
 ** ANGRYBIRDS AI AGENT FRAMEWORK
 ** Copyright (c) 2014,XiaoYu (Gary) Ge, Stephen Gould, Jochen Renz
 **  Sahan Abeyasinghe, Jim Keys,   Andrew Wang, Peng Zhang
 ** All rights reserved.
**This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**To view a copy of this license, visit http://www.gnu.org/licenses/
 *****************************************************************************/
package ab.vision;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.List;

public class Vision {
	private BufferedImage image;
	private VisionMBR visionMBR = null;
	private VisionRealShape visionRealShape = null;
	
	public Vision(BufferedImage image)
	{
		this.image = image;
	}
	
	public List<ABObject> findBirdsMBR()
	{
		if (visionMBR == null)
		{
			visionMBR = new VisionMBR(image);
		} 
		return visionMBR.findBirds();
			
	}
	/**
	 * @return a list of MBRs of the blocks in the screenshot. Blocks: Stone, Wood, Ice
	 * */
	public List<ABObject> findBlocksMBR()
	{
		if (visionMBR == null)
		{
			visionMBR = new VisionMBR(image);
		}
		return visionMBR.findBlocks();
	}
	
	public List<ABObject> findTNTs()
	{
		if(visionMBR == null)
		{
			visionMBR = new VisionMBR(image);
		}
		return visionMBR.findTNTs();
	}
	public List<ABObject> findPigsMBR()
	{
		if (visionMBR == null)
		{
			visionMBR = new VisionMBR(image);
		}
		return visionMBR.findPigs();
	}
	public List<ABObject> findPigsRealShape()
	{
		if(visionRealShape == null)
		{
			visionRealShape = new VisionRealShape(image);
		}
		
		return visionRealShape.findPigs();
	} 
	public List<ABObject> findBirdsRealShape()
	{
		if(visionRealShape == null)
		{
			visionRealShape = new VisionRealShape(image);
		}
		
		return visionRealShape.findBirds();
	}
	
	public List<ABObject> findHills()
	{
		if(visionRealShape == null)
		{
			visionRealShape = new VisionRealShape(image);
		}
		
		return visionRealShape.findHills();
	} 
	
	
	public Rectangle findSlingshotMBR()
	{
		if (visionMBR == null)
		{
			visionMBR = new VisionMBR(image);
		}
		return visionMBR.findSlingshotMBR();
	}
	public List<Point> findTrajPoints()
	{
		if (visionMBR == null)
		{
			visionMBR = new VisionMBR(image);
		}
		return visionMBR.findTrajPoints();
	}
	/**
	 * @return a list of real shapes (represented by Body.java) of the blocks in the screenshot. Blocks: Stone, Wood, Ice 
	 * */
	public List<ABObject> findBlocksRealShape()
	{
		if(visionRealShape == null)
		{
			visionRealShape = new VisionRealShape(image);
		}
		List<ABObject> allBlocks = visionRealShape.findObjects();
		
		return allBlocks;
	}
	public VisionMBR getMBRVision()
	{
		if(visionMBR == null)
			visionMBR = new VisionMBR(image);
		return visionMBR;
	}
}
