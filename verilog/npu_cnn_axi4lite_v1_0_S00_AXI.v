`timescale 1 ns / 1 ps

module npu_cnn_axi4lite_v1_0_S00_AXI #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6,
    parameter DATA_WIDTH = 16,            // for image/weights
    parameter BRAM_ADDR_WIDTH = 14        // 16K words (can be increased)
)(
    input wire S_AXI_ACLK,
    input wire S_AXI_ARESETN,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input wire [2:0] S_AXI_AWPROT,
    input wire S_AXI_AWVALID,
    output wire S_AXI_AWREADY,
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input wire S_AXI_WVALID,
    output wire S_AXI_WREADY,
    output wire [1:0] S_AXI_BRESP,
    output wire S_AXI_BVALID,
    input wire S_AXI_BREADY,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input wire [2:0] S_AXI_ARPROT,
    input wire S_AXI_ARVALID,
    output wire S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output wire [1:0] S_AXI_RRESP,
    output wire S_AXI_RVALID,
    input wire S_AXI_RREADY,
    output reg irq
);

//--------------------------------------------
// AXI4-Lite Registers
//--------------------------------------------
localparam REG_CTRL      = 6'h00; // [0]=start [1]=reset [2]=irq_en
localparam REG_STATUS    = 6'h01; // [0]=done [1]=busy
localparam REG_IMG_BASE  = 6'h02; // BRAM region base for image
localparam REG_OUT_BASE  = 6'h03; // ...for output
localparam REG_WGT_BASE  = 6'h04; // ...for weights
localparam REG_IMG_SIZE  = 6'h05; // [15:8]=H, [7:0]=W
localparam REG_CHAN_INFO = 6'h06; // [15:8]=InC, [7:0]=OutC
localparam REG_KER_SIZE  = 6'h07; // [15:8]=K_H, [7:0]=K_W
localparam REG_NUM_LAYERS= 6'h08; // Number of layers

reg [31:0] reg_ctrl, reg_status;
reg [31:0] reg_img_base, reg_out_base, reg_wgt_base;
reg [31:0] reg_img_size, reg_chan_info, reg_ker_size, reg_num_layers;

//--------------------------------------------
// Block RAM buffer (expand as needed)
//--------------------------------------------
reg [DATA_WIDTH-1:0] bram [0:(1<<BRAM_ADDR_WIDTH)-1];

// Fix: Declare idx ONCE at module scope, not inside any always/case
integer idx;

//--------------------------------------------
// AXI4-Lite Write FSM
//--------------------------------------------
reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
reg axi_awready, axi_wready, axi_bvalid;
reg [1:0] axi_bresp;
reg aw_en;
assign S_AXI_AWREADY = axi_awready;
assign S_AXI_WREADY  = axi_wready;
assign S_AXI_BVALID  = axi_bvalid;
assign S_AXI_BRESP   = axi_bresp;

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        axi_awready <= 1'b0; aw_en <= 1'b1;
        axi_wready  <= 1'b0;
        axi_bvalid  <= 1'b0;
        axi_bresp   <= 2'b00;
    end else begin
        if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
            axi_awready <= 1'b1; aw_en <= 1'b0; axi_awaddr <= S_AXI_AWADDR;
        end else if (S_AXI_BREADY && axi_bvalid) begin
            aw_en <= 1'b1; axi_awready <= 1'b0;
        end else begin
            axi_awready <= 1'b0;
        end
        if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en)
            axi_wready <= 1'b1;
        else
            axi_wready <= 1'b0;
        // Register/buffer write
        if (axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID) begin
            case(axi_awaddr[5:2])
                REG_CTRL:      reg_ctrl      <= S_AXI_WDATA;
                REG_IMG_BASE:  reg_img_base  <= S_AXI_WDATA;
                REG_OUT_BASE:  reg_out_base  <= S_AXI_WDATA;
                REG_WGT_BASE:  reg_wgt_base  <= S_AXI_WDATA;
                REG_IMG_SIZE:  reg_img_size  <= S_AXI_WDATA;
                REG_CHAN_INFO: reg_chan_info <= S_AXI_WDATA;
                REG_KER_SIZE:  reg_ker_size  <= S_AXI_WDATA;
                REG_NUM_LAYERS:reg_num_layers<= S_AXI_WDATA;
                default: begin
                    if (axi_awaddr[5:2] >= 6'h10) begin
                        idx = (axi_awaddr[5:2]-6'h10)*4;
                        if (S_AXI_WSTRB[0]) bram[idx+0] <= S_AXI_WDATA[15:0];
                        if (S_AXI_WSTRB[1]) bram[idx+1] <= S_AXI_WDATA[31:16];
                    end
                end
            endcase
            axi_bvalid <= 1'b1;
            axi_bresp  <= 2'b00;
        end else if (axi_bvalid && S_AXI_BREADY) begin
            axi_bvalid <= 1'b0;
        end
    end
end

//--------------------------------------------
// AXI4-Lite Read FSM
//--------------------------------------------
reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
reg axi_arready, axi_rvalid;
reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
reg [1:0] axi_rresp;
assign S_AXI_ARREADY = axi_arready;
assign S_AXI_RVALID  = axi_rvalid;
assign S_AXI_RDATA   = axi_rdata;
assign S_AXI_RRESP   = axi_rresp;

always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        axi_arready <= 1'b0; axi_araddr <= 0;
        axi_rvalid  <= 1'b0;
        axi_rdata   <= 0;
        axi_rresp   <= 2'b00;
    end else begin
        if (~axi_arready && S_AXI_ARVALID) begin
            axi_arready <= 1'b1; axi_araddr <= S_AXI_ARADDR;
        end else begin
            axi_arready <= 1'b0;
        end
        if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
            case (S_AXI_ARADDR[5:2])
                REG_CTRL:      axi_rdata <= reg_ctrl;
                REG_STATUS:    axi_rdata <= reg_status;
                REG_IMG_BASE:  axi_rdata <= reg_img_base;
                REG_OUT_BASE:  axi_rdata <= reg_out_base;
                REG_WGT_BASE:  axi_rdata <= reg_wgt_base;
                REG_IMG_SIZE:  axi_rdata <= reg_img_size;
                REG_CHAN_INFO: axi_rdata <= reg_chan_info;
                REG_KER_SIZE:  axi_rdata <= reg_ker_size;
                REG_NUM_LAYERS:axi_rdata <= reg_num_layers;
                default: begin
                    if (S_AXI_ARADDR[5:2] >= 6'h10) begin
                        idx = (S_AXI_ARADDR[5:2]-6'h10)*4;
                        axi_rdata[15:0]  <= bram[idx+0];
                        axi_rdata[31:16] <= bram[idx+1];
                    end else
                        axi_rdata <= 0;
                end
            endcase
            axi_rvalid <= 1'b1; axi_rresp <= 2'b00;
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
        end
    end
end

//--------------------------------------------
// NPU Control FSM (STUB: implement CNN logic here)
//--------------------------------------------
reg running;
always @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
        reg_status   <= 0;
        irq          <= 0;
        running      <= 0;
    end else begin
        if (reg_ctrl[1]) begin // reset
            reg_status   <= 0;
            running      <= 0;
        end else if (reg_ctrl[0] && !running) begin // start
            running    <= 1;
            reg_status <= 2'b10; // busy
            // --- Insert CNN compute FSM here ---
        end else if (running) begin
            // Placeholder: replace with compute-done condition
            running    <= 0;
            reg_status <= 2'b01; // done
            if (reg_ctrl[2]) irq <= 1;
        end else begin
            if (irq) irq <= 0;
        end
    end
end

endmodule
